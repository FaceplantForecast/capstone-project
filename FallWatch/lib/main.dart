import 'package:flutter/material.dart';
import 'dart:io' show Platform;
import 'firebase_options.dart';
import 'pages/login_page.dart';
import 'pages/fall_history_page.dart';
import 'pages/graph_page.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'services/notification_services.dart';
import 'services/auth_gate.dart';
import 'widgets/AppLockObserver.dart';



void main() async {
  WidgetsFlutterBinding.ensureInitialized();
   
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  if (!Platform.isWindows) { 
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
    NotificationService.initialize();
  } else {
    print('Skipping FCM messaging setup on Windows');
  }

  WidgetsBinding.instance.addObserver(AppLockObserver.instance);

  
  runApp(const MyApp());
}

Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
  print("🔥 Background message received: ${message.notification?.title}");
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FallWatch',
      debugShowCheckedModeBanner: false,
      home: const AuthGate(),
      routes: {
        '/login': (context) => const LoginPage(title: 'Login Page'),
        '/fall_history': (context) => const FallHistoryPage(),
        '/graph_page': (context) => const GraphPage(),
      },
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueAccent),
        scaffoldBackgroundColor: const Color(0xFFF4F6FA),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 20,
            color: Colors.black87,
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        labelStyle: const TextStyle(fontSize: 16),
        prefixIconColor: Colors.blueAccent,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
          backgroundColor: Colors.white,
          textStyle: const TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ),
    ),
  );
  }
}