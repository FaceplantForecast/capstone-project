// import 'dart:io' show Platform;
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
// import 'package:local_auth/local_auth.dart';
// import '../services/user_settings.dart';
import '../widgets/SplashScreen.dart';
// import '../pages/user_page.dart';
import '../pages/login_page.dart';
import '../widgets/AppLockGate.dart';
// import '../services/biometric_settings.dart';


class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        // Still determining auth state
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const SplashScreen();
        }

        // Not logged in
        if (!snapshot.hasData) {
          return const LoginPage(title: 'Login');
        }

        // Logged in
        return const AppLockGate();
      },
    );
  }
//   const AuthGate ({super.key});

//   @override
//   State<AuthGate> createState() => _AuthGateState();
// }

// class _AuthGateState extends State<AuthGate> {
//   final _auth = FirebaseAuth.instance;
//   final _localAuth = LocalAuthentication();
//   final user = FirebaseAuth.instance.currentUser;

//   @override
//   void initState() {
//     super.initState();
//     WidgetsBinding.instance.addPostFrameCallback((_) {
//       _checkAuth();
//     });
//   }

//   Future<void> _checkAuth() async {
//     try {
//       print("Starting auth checks");
//       final user = _auth.currentUser;

//       // Not logged in
//       if (user == null) {
//         print('Login, user is null');
//         _goToLogin();
//         return;
//       }
//       //Windows not supported yet
//       if (Platform.isWindows) {
//         print('Is windows, skipping');
//         _goToHome();
//         return;
//       }
      
//       //biometrics enabled check
//       final settings = await UserSettings.getSettings();
//       final biometricsEnabled = settings['biometricsEnabled'] ?? false;
//       final timeout = settings['sessionTimeoutMinutes'] ?? 5;

//       if (!biometricsEnabled) {
//         _goToHome();
//         return;
//       }

//       //Session timeouts
//       final expired = await SecureSession.isExpired(timeout);
//       print("Test here " + expired.toString());
//       if (!expired) {
//         _goToHome();
//         return;
//       }

//       //Check device support
//       final canAuth =
//         await _localAuth.canCheckBiometrics ||
//         await _localAuth.isDeviceSupported();

//       if (!canAuth) {
//         print('Device not supported/not checked for biometrics');
//         _goToLogin();
//         return;
//       }

//       //Prompt Biometrics
//       final success = await _localAuth.authenticate(
//         localizedReason: 'Please Authenticate to access FallWatch',
//         biometricOnly: false,
//         persistAcrossBackgrounding: true,
//       );

//       if (success) {
//         await SecureSession.refresh();
//         _goToHome();
//       } else {
//         _goToLogin();
//       }
//     } catch (e) {
//       print('Biometric auth error: $e');
//       _goToLogin();
//     }
//   }
//   void _goToHome() {
//     final email = user?.email ?? '';
//     final userNick = email.contains('@')
//         ? email.split('@').first
//         : 'User';
//     if (!mounted) return;
//     Navigator.pushReplacement(
//       context,
//       MaterialPageRoute(
//         builder: (_) => 
//           UserPage(userNick: userNick)),
//     );
//   }
//   void _goToLogin() {
//     if (!mounted) return;
//     Navigator.pushReplacement(
//       context,
//       MaterialPageRoute(
//         builder: (_) =>
//             const LoginPage(title: 'Login'),
//       ),
//     );
//   }
//   @override
//   Widget build(BuildContext context) { 
//     return StreamBuilder(
//       stream: FirebaseAuth.instance.authStateChanges(),
//       builder: (context, snapshot) {
//         // Still checking
//         if (snapshot.connectionState == ConnectionState.waiting) {
//           return const SplashScreen();
//         }

//         // Not logged in → send to login
//         if (!snapshot.hasData) {
//           return const LoginPage(title: 'Login');
//         }

//         // Logged in → check biometric lock
//         return const AppLockGate();
//       },
//     );
//   }
}

