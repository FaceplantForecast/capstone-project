import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:audioplayers/audioplayers.dart';
import 'fall_logger.dart';
// import 'package:flutter/material.dart';
// @pragma('vm:entry-point')
// Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
//   final plugin = FlutterLocalNotificationsPlugin();
//   await plugin.initialize(
//     const InitializationSettings(
//       android: AndroidInitializationSettings('@mipmap/ic_launcher'),
//     ),
//   );
//   await plugin.show(
//     message.hashCode,
//     message.data['title'] ?? message.notification?.title ?? 'Fall Detected!',
//     message.data['body']  ?? message.notification?.body  ?? 'A fall was detected.',
//     const NotificationDetails(
//       android: AndroidNotificationDetails(
//         NotificationService._channelId,
//         NotificationService._channelName,
//         importance: Importance.max,
//         priority: Priority.high,
//         playSound: true,
//         // Vibration pattern: wait 0ms, vibrate 500ms, wait 200ms, vibrate 500ms
//         enableVibration: true,
//       ),
//     ),
//   );
 
//   print('📩 Background FCM handled: ${message.notification?.title}');
// }

class NotificationService {
  static final FlutterLocalNotificationsPlugin _localNotifications =
    FlutterLocalNotificationsPlugin();

  static final AudioPlayer _audioPlayer = AudioPlayer();

  //channels and files

  static void initialize() async {
    const AndroidInitializationSettings androidInit =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const InitializationSettings initSettings =
        InitializationSettings(android: androidInit);

    await _localNotifications.initialize(initSettings);

    // create android channel

    FirebaseMessaging.onMessage.listen((RemoteMessage message) async {
      print('📩 Message received in foreground: ${message.notification?.title}');
      await _showNotification(message);

      final data = message.data;
      final fallDetected = data['fall_detected'] == '1' || data['fall_detected'] == 'true';
      await FallLogger.logFallEvent(
        source: 'FCM',
        fallDetected: fallDetected,
        timestamp: DateTime.now(),
        extra: data,
      );
    });
  }

  static Future<void> _showNotification(RemoteMessage message) async {
    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      'high_importance_channel',
      'Fall Detection Alerts',
      importance: Importance.max,
      priority: Priority.high,
      playSound: true,
 //     sound: RawResourceAndroidNotificationSound('ouch.mp3'),
    );

    const NotificationDetails details =
        NotificationDetails(android: androidDetails);

    await _localNotifications.show(
      0,
      message.notification?.title ?? 'No Title',
      message.notification?.body ?? 'No Body',
      details,
    );
  }

  static Future<void> showManualNotification({
    required String title,
    required String body,
  }) async {
    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      'high_importance_channel',
      'Fall Detection Alerts',
      importance: Importance.max,
      priority: Priority.high,
      playSound: true,
     // sound: RawResourceAndroidNotificationSound('ouch.mp3'),
    );

    const NotificationDetails details =
        NotificationDetails(android: androidDetails);

    await _localNotifications.show(1, title, body, details);
  }
}