import 'dart:async';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'fall_logger.dart';
import 'notification_services.dart';

class DeviceMonitorService {
  static final DeviceMonitorService _instance =
      DeviceMonitorService._internal();

  factory DeviceMonitorService() => _instance;

  DeviceMonitorService._internal();

  // channel
  static const String server = 'gcr-ws-482782751069.us-central1.run.app';
  static const String token = 'M8b4eFJHq2pI3V9nW5r0dE-PLZpQyX7uB1cTa9kN4mE';
  static final String wsUrl = 'wss://$server?token=$token';

  WebSocketChannel? _channel;
  Timer? _timeElapsed;
  Timer? _ping;
  Timer? _reconnectTimer;
  bool _manuallyClosed = false;

  bool connected = false;
  bool isWallPower = true;
  bool fallDetected = false;
  bool RadarOnline = false;
  bool PiOnline = false;
  DateTime? lastTimestamp;
  DateTime? powerStateChanged;

  final StreamController<void> _stateController =
    StreamController.broadcast(sync: true);

  Stream<void> get stateStream => _stateController.stream;

  void start() {
    if (_channel != null) return; //alredy running
    _connect();

    _timeElapsed ??= Timer.periodic(const Duration(seconds: 1), (_) 
      => _stateController.add(connected),
    );
  }

  void _connect() {
    _manuallyClosed = false;

    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      _ping?.cancel();
      _ping = Timer.periodic(const Duration(seconds: 10), (_) {
        try {
         _channel?.sink.add('ping');
        } catch (_) {}
      },
      );

      _channel!.stream.listen(
        _handleMessage,
        onError: (_) => _scheduleReconnect(),
        onDone: () => _scheduleReconnect(),
      );
    } catch (_) {
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    if (_manuallyClosed) return;

    connected = false;
    _stateController.add(null);

    _ping?.cancel();

    _reconnectTimer?.cancel();
    _reconnectTimer =
      Timer(const Duration(seconds: 2), _connect);
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
    fallDetected = false;
    _stateController.add(null);
  }

  void _handleMessage(dynamic data) async {
    print('Received WebSocket message: $data');
    connected = true;
    _stateController.add(null);

    final s = data.toString().trim();
    print('Processing message: $s');
    final jsonString =
      s.contains("'") ? s.replaceAll("'", '"') : s;

    Map<String, dynamic> msg;
    print('Decoding JSON: $jsonString');
    try {
      msg = json.decode(jsonString);
      print('Decoded message: $msg');
    } catch (_) {
      return;
    }

    final payload = msg['payload'];
    if (payload is! Map) return;

    // fall status
    final fdRaw = payload['fall_detected'];
    fallDetected = fdRaw == 1 ||
        fdRaw == true ||
        fdRaw == '1' ||
        fdRaw == 'true';

    lastTimestamp = DateTime.now();
    final fallSource = payload['device_id'] ?? payload['device'] ?? 'WebSocket';
    print(fallSource);
    if (fallDetected) {
      await FallLogger.logFallEvent(
        source: fallSource,
        fallDetected: true,
        timestamp: lastTimestamp!,
        extra: msg,
      );

      await NotificationService.showManualNotification(
        title: 'Fall Detected!',
        body: 'A fall was detected.',
      );
    }
    // device_id (role): (sim-pi-0#)

    /* [data] {
          "type": "status",
          "event": "boot_connected",
          "status": "connected",
          "device": "sim-pi-01",
          "timestamp": "03-19-2026 11:57:28"
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
      final status = payload['status']?.toString() ?? '';
      final newState = status.contains('Wall Power');
      print(status);
      isWallPower = status.contains('Wall Power');
      if (newState != isWallPower) {
        powerStateChanged = DateTime.now();
      }
      if (status.contains('Wall Power')) {
        isWallPower = true;
      } else if (status.contains('Battery')) {
        isWallPower = false;
      }
      isWallPower = newState;
    }

    _stateController.add(null);
  }

  void stop() {
    _manuallyClosed = true;
    _ping?.cancel();
    _reconnectTimer?.cancel();
    _timeElapsed?.cancel();
    _channel?.sink.close();
    _channel = null;
    connected = false;
  }
}