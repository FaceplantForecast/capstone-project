import 'dart:async';
import 'dart:convert';
import 'dart:ui';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'fall_logger.dart';
import 'notification_services.dart';

enum DeviceStatus {
  monitoring,
  calibrating,
  calibrationFailed,
  systemFailure,
  disconnected,
}

extension DeviceStatusLabel on DeviceStatus {
  String get label {
    switch (this) {
      case DeviceStatus.monitoring:
        return "Monitoring...";
      case DeviceStatus.calibrating:
        return "Calibrating...";
      case DeviceStatus.calibrationFailed:
        return "Calibration Failed";
      case DeviceStatus.systemFailure:
        return "System Failure";
      case DeviceStatus.disconnected:
        return "Offline";
    }
  }
}

class DeviceState {
  final String deviceId;

  bool connected;
  bool isWallPower;
  String? accountId;
  DateTime? lastSeen;
  DateTime? batteryStartTime; //when device switched to battery
  String? displayName; // optional device name
  DeviceStatus status;

  Timer? _calibrationTimer; // tracks time since boot for calibration timeout

  DeviceState({
    required this.deviceId,
    this.connected = false,
    this.isWallPower = true,
    this.accountId,
    this.lastSeen,
    this.batteryStartTime,
    this.displayName,
    this.status = DeviceStatus.disconnected,
  });

  String get label => (displayName != null && displayName!.isNotEmpty) ? displayName! : deviceId;

  void startCalibrationTimer(VoidCallback onTimeout) {
    _calibrationTimer?.cancel();
    _calibrationTimer = Timer(const Duration(seconds: 60), () async {
      if (status == DeviceStatus.calibrating) {
        status = DeviceStatus.calibrationFailed;
        onTimeout();
        await Future.delayed(const Duration(seconds: 15));  
        status = DeviceStatus.disconnected;
        onTimeout();      
      }
    });
  }
 
  void cancelCalibrationTimer() {
    _calibrationTimer?.cancel();
    _calibrationTimer = null;
  }


  Map<String, dynamic> toFirestoreStatus() => {
    'deviceId': deviceId,
    'connected': connected,
    'isWallPower': isWallPower,
    'accountId': accountId,
    'status': status.name,
    'lastSeen': lastSeen != null ? Timestamp.fromDate(lastSeen!) : null,
    'batteryStartTime': batteryStartTime != null ? Timestamp.fromDate(batteryStartTime!) : null,
  };

  Map<String, dynamic> toFirestoreNew() => {
    ...toFirestoreStatus(),
    if (displayName != null) 'displayName': displayName,
  };

  factory DeviceState.fromFirestore(Map<String, dynamic> data) {
    DateTime? parseTs(String key) {
      final v = data[key];
      if (v is Timestamp) return v.toDate();
      if (v is String) return DateTime.tryParse(v);
      return null;
    }

    DeviceStatus parseStatus(String? s) {
      return DeviceStatus.values.firstWhere(
        (e) => e.name == s,
        orElse: () => DeviceStatus.disconnected,
      );
    }

    return DeviceState(
      deviceId: data['deviceId'] as String,
      connected: data['connected'] as bool? ?? false,
      isWallPower: data['isWallPower'] as bool? ?? true,
      accountId: data['accountId'] as String?,
      displayName: data['displayName'] as String?,
      lastSeen: parseTs('lastSeen'),
      batteryStartTime: parseTs('batteryStartTime'),
      status: parseStatus(data['status'] as String?),
      
    );
  }
}



class DeviceMonitorService {
  static final DeviceMonitorService _instance =
      DeviceMonitorService._internal();

  factory DeviceMonitorService() => _instance;

  DeviceMonitorService._internal();
  
  // FIRESTORE DB
  final _db = FirebaseFirestore.instance;
  CollectionReference get _devicesRef => _db.collection('devices');
  StreamSubscription? _firestoreSub;

  // channel
  static const String server = 'gcr-ws-482782751069.us-central1.run.app';
  static const String token = 'M8b4eFJHq2pI3V9nW5r0dE-PLZpQyX7uB1cTa9kN4mE';
  static final String wsUrl = 'wss://$server?token=$token';

  final Map<String, DeviceState> devices = {};


  WebSocketChannel? _channel;
  Timer? _timeElapsed;
  Timer? _ping;
  Timer? _reconnectTimer;
  bool _manuallyClosed = false;

  Timer? _reNotifyTimer;

  bool connected = false;
  bool fallDetected = false;
  DateTime? lastTimestamp;
  String? fallSourceDevice; // for location display on fall

  final StreamController<void> _stateController =
    StreamController.broadcast(sync: true);

  Stream<void> get stateStream => _stateController.stream;

  void start() {
    _subscribeToFirestore();
    if (_channel != null) return; //alredy running
    _connect();
    _timeElapsed ??= Timer.periodic(const Duration(seconds: 1), (_) 
      => _stateController.add(null),
    );
  }

  void _subscribeToFirestore() {
    _firestoreSub?.cancel();
    _firestoreSub = _devicesRef.snapshots().listen((snapshot) {
      for (final change in snapshot.docChanges) {
        final data = change.doc.data() as Map<String, dynamic>?;
        if (data == null) continue;

        final deviceId = data['deviceId'] as String?;
        if (deviceId == null) continue;

        if (change.type == DocumentChangeType.removed) {
          devices.remove(deviceId);
        } else {
          final existing = devices[deviceId];
          final updated = DeviceState.fromFirestore(data);

          if (existing != null) {
            existing.connected = updated.connected;
            existing.isWallPower = updated.isWallPower;
            existing.accountId = updated.accountId;
            existing.lastSeen = updated.lastSeen;
            existing.batteryStartTime = updated.batteryStartTime;
            existing.displayName = updated.displayName;
          } else {
            devices[deviceId] = updated;
          }
        }
      }
      _stateController.add(null);
    }, onError: (e) {
      print('Firestore Listener Error (Devices): $e');
    });
  }

  // this allows the displayname to remain through updates
  Future<void> _upsertDevice(DeviceState device,
      {bool isNew = false}) async {
    final ref = _devicesRef.doc(device.deviceId);
    if (isNew) {
      await ref.set(device.toFirestoreNew(), SetOptions(merge: true));
    } else {
      await ref.set(device.toFirestoreStatus(), SetOptions(merge: true));
    }
  }

  Future<void> renameDevice(String deviceId, String newName) async {
    await _devicesRef.doc(deviceId).update({'displayName': newName});
    devices[deviceId]?.displayName = newName;
    _stateController.add(null);
  }

  void _sendToServer(Map<String, dynamic> message) {
    if (_channel == null) {
      print("WebSocket channel is not initialized.");
      return;
    }
    try {
      _channel!.sink.add(jsonEncode(message));
    } catch (e) {
      print("Error sending message to server: $e");
    }
  }

  void _connect() {
    _manuallyClosed = false;
    if (_channel != null) return; 

    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      _ping?.cancel();
      _ping = Timer.periodic(const Duration(seconds: 10), (_) {
      try {
        _channel?.sink.add('ping');
      } catch (_) {}
      });

      _channel!.stream.listen(
        _handleMessage,
        onError: (e) {
          print("WS Error: $e");
          _scheduleReconnect();
        },
        onDone: () {
          print("WS Closed by server");
          _scheduleReconnect();
        },
      );
    } catch (_) {
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    if (_manuallyClosed) return;

    try {
      _channel?.sink.close();
    } catch (_) {}
    _channel = null;

    connected = false;
    _stateController.add(null);

    _ping?.cancel();

    _reconnectTimer?.cancel();
    _reconnectTimer =
      Timer(const Duration(seconds: 1), _connect);
    print('Reconnecting to WebSocket');
  }
  /*
  {
  "type": "hello",
  "role": "app",
  "ts": 1773937292945
}
[data] {
  "type": "status",
  "event": "boot_connected",
  "status": "connected",
  "device": "raspberry_pi",
  "timestamp": "03-19-2026 11:21:48"
}
[data] {
  "type": "status",
  "event": "power_state",
  "status": "System on Wall Power",
  "device": "raspberry_pi",
  "timestamp": 1773937315.5071387
} */

  void acknowledgeFall() {
    _reNotifyTimer?.cancel();
    fallDetected = false;
    _stateController.add(null);

    // stream to server for acknowledgement
    final uid = FirebaseAuth.instance.currentUser?.uid ?? 'unknown user';
    final email = FirebaseAuth.instance.currentUser?.email ?? 'unknown email';
    _sendToServer({
        'event': 'acknowledge_fall',
        'acknowledgedBy': uid,
        'acknowledgedByEmail': email,
        'deviceId': fallSourceDevice,
        'timestamp': DateTime.now().toIso8601String(),
      }
    );
  }

  void _handleRemoteAck(Map<String, dynamic> payload) {
    _reNotifyTimer?.cancel();
    fallDetected = false;
    fallSourceDevice = null;
    final by = payload['acknowledgedByEmail'] ?? payload['acknowledgedBy'] ?? 'unknown user';
    print('Fall acknowledged by $by from server command');
    _stateController.add(null);
  }

  void _startReNotifyTimer() {
    _reNotifyTimer?.cancel();
    _reNotifyTimer = Timer.periodic(const Duration(seconds: 15), (_) async {
      if (!fallDetected) {
        _reNotifyTimer?.cancel();
        return;
      }
      final source = fallSourceDevice ?? 'Unknown Device';
      await NotificationService.showManualNotification(
        title: 'Urgent! Fall Still Detected!',
        body: 'A fall is still detected on $source!',
      );
    });
  }

  // device renaming
  // void renameDevice(String deviceId, String newName) {
  //   final device = devices[deviceId];
  //   if (device == null) return;
  //   device.displayName = newName;
  //   _saveDevices();
  //   _stateController.add(null);
  // }

  void _handleMessage(dynamic data) async {
    connected = true;
    _stateController.add(null);

    final s = data.toString().trim();
    final jsonString =
      s.contains("'") ? s.replaceAll("'", '"') : s;

    Map<String, dynamic> msg;
    try {
      msg = json.decode(jsonString);
      print('Decoded message: $msg');
    } catch (_) {
      return;
    }
    if (msg['type'] == 'hello') {
      print('WebSocket connection established');
      return;
    }
    final payload = msg['payload'];
    final internalPayload = payload['payload'];
    print(payload);
    if (payload is! Map) return;

    if (payload['event'] == 'acknowledge_fall') {
      _handleRemoteAck(internalPayload ?? payload);
      return;
    }

    /* {type: data, payload: {type: status, event: boot_connected, status: disconnected, device: sim-pi-01, timestamp: 1774409482.1428337}} */

    // add new device if boot connected
    if (payload['event'] == 'boot_connected') {
      // device_status = "Calibrating...";
      final deviceId = payload['device'];
      if (payload['status'] == 'disconnected') {
        final device = devices[deviceId];
        if (device != null) {
          print('device boot disconnected: $deviceId');
          device.connected = false;
          device.status = DeviceStatus.disconnected;
          device.cancelCalibrationTimer();
          device.lastSeen = DateTime.now();
          await _upsertDevice(device);
          _stateController.add(null);
        }
        return;
      }
      if (payload['status'] == 'connected') {
        print('Device Boot Connected: $deviceId');
      final device = devices.putIfAbsent(
        deviceId,
        () => DeviceState(deviceId: deviceId)
      );

      device.connected = true;
      device.lastSeen = DateTime.now();
      device.accountId = payload['account_id'];
      device.status = DeviceStatus.calibrating;

      device.startCalibrationTimer(() async {
        await _upsertDevice(device);
        _stateController.add(null);
        print('Calibration timeout for device $deviceId');
      });

      await _upsertDevice(device, isNew: true);
      _stateController.add(null);
      }
    }



    if (payload['msg_type'] == 'system_event' &&
        internalPayload != null) {
      final deviceId = payload['device_id'] as String? ?? payload['device'] as String?;
      final device = deviceId != null ? devices[deviceId] : null;
 
      final message    = internalPayload['message'] as String? ?? '';
      final eventType  = internalPayload['event_type'] as String? ?? '';
 
      if (message == 'finished background subtraction' ||
          eventType == 'calibration_complete') {
        if (device != null) {
          device.cancelCalibrationTimer();
          device.status   = DeviceStatus.monitoring;
          device.lastSeen = DateTime.now();
          await _upsertDevice(device);
          _stateController.add(null);
          print('$deviceId finished calibrating');
        }
      }
 
      if (eventType == 'process_restart_failed') {
        if (device != null) {
          device.cancelCalibrationTimer();
          device.status   = DeviceStatus.systemFailure;
          device.lastSeen = DateTime.now();
          await _upsertDevice(device);
          _stateController.add(null);
          print('SYSTEM FAILURE on $deviceId');
        }
      }
    }
    
    // {type: data, payload: {msg_type: system_event, ts_send: 04-09-2026 12:19:25, device_id: raspberry_pi, account_id: account-1, payload: {event_type: status, message: finished background subtraction}}}

    // fall event
    if (payload['msg_type'] == 'fall_event') {
      final fallDevice = internalPayload['device_id'] ?? payload['device_id'] ?? 'Unknown WebSocket';
      print('Source of Fall: $fallDevice');
      
      final device = devices[fallDevice];
      if (device == null) {
        print("Unrecognized device for fall event: $fallDevice");
        return;
      }
      print(fallDevice);
      device.accountId = payload['account_id'];
      device.lastSeen = DateTime.now();
      await _upsertDevice(device);
    }

    // fall status
    final fdRaw = internalPayload?['fall_detected'];
    print('Fall or not? $fdRaw');
    fallDetected = 
      (fdRaw) == 1 ||
      (fdRaw) == true ||
      (fdRaw) == '1' ||
      (fdRaw) == 'true';

    lastTimestamp = DateTime.now();
    final fallSource = payload['device_id'] ?? payload['device'] ?? 'Unknown WebSocket';
    
    if (fallDetected) {
      fallSourceDevice = fallSource;
      print(fallSourceDevice);
      await FallLogger.logFallEvent(
        source: fallSource,
        fallDetected: true,
        timestamp: lastTimestamp!,
        extra: msg,
      );

      await NotificationService.showManualNotification(
        title: 'Fall Detected!',
        body: 'A fall was detected on ${devices[fallSource]?.label ?? fallSource}.',
      );
      // Start the ack loop
      _startReNotifyTimer();
    }

    // device_id (role): (sim-pi-0#)

    /* [data] {
          "type": "status",
          "event": "boot_connected",
          "status": "connected",
          "device": "sim-pi-01",
          "timestamp": "03-19-2026 11:57:28"
        } 
        [data] {
          "type": "status",
          "event": "power_state",
          "status": "System on Wall Power",
          "device": "sim-pi-02",
          "timestamp": 1775156082.8368368
        } */
// for simulated pi boot
    
 /* [data] {
  "msg_type": "fall_event",
  "ts_send": "03-19-2026 11:25:33",
  "payload": {
    "fall_detected": 1,
    "probability": 0.87,
    "frame_id": 712108,
    "ts_fall": "03-19-2026 11:25:33",
    "device_id": "sim-pi-01",
    "account_id": "account-1"
  }
} */

    // power status
    if (payload['event'] == 'power_state') {
      final deviceId = payload['device_id'] ?? payload['device'] ?? 'NOT FOUND';
      final device = devices[deviceId];
      if (device == null) {
        print("Unrecognized device for power status: $deviceId");
        return;
      }
      final status = payload['status'] as String? ?? '';
      print('Power Status for $deviceId: $status');
      final wasWallPower = device.isWallPower;
      print(wasWallPower);
      device.isWallPower = status.contains('Wall Power');
      device.lastSeen = DateTime.now();

      if (!device.isWallPower && wasWallPower) {
        device.batteryStartTime = DateTime.now();
      } else if (!wasWallPower && device.isWallPower) {
        device.batteryStartTime = null;
      }

      await _upsertDevice(device);
      print('Device $deviceId power status: ${device.isWallPower ? 'Wall Power' : 'Battery'}');
    }
    _stateController.add(null);
  }

  

 // event_type: process_restart_failed (for complete failure)

 // boot -> Calibrating... -> System FAILURE or Monitoring... 
void sendRecalibrationCommand(String deviceId) {
  print("Channel exists? ${_channel != null}");
  if (_channel == null) {
    print("WebSocket channel is not initialized.");
    return;
  }
  _sendToServer({
    'command': 'redo_background_scan',
    'deviceId': deviceId,
    'timestamp': DateTime.now().toIso8601String(),
  });
  print('Sent recalibration command for device $deviceId');
  final device = devices[deviceId];
  if (device != null) {
    device.status = DeviceStatus.calibrating;
    device.startCalibrationTimer(() async {
      await _upsertDevice(device);
      _stateController.add(null);
      print('Calibration timeout for device $deviceId');
    });
  }
}

  void stop() {
    _manuallyClosed = true;
    _firestoreSub?.cancel();
    _ping?.cancel();
    _reconnectTimer?.cancel();
    _timeElapsed?.cancel();
    _reNotifyTimer?.cancel();
    for (final d in devices.values) {
      d.cancelCalibrationTimer();
    }
    _channel?.sink.close();
    _channel = null;
    connected = false;
  }
}