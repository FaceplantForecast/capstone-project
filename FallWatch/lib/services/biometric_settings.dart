import 'package:flutter_secure_storage/flutter_secure_storage.dart';
// import 'package:firebase_auth/firebase_auth.dart';
// import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SecureSession {
  static const _storage = FlutterSecureStorage();
  static const _key = 'last_auth_time';
  static const _hasLoggedInKey = 'has_logged_in_once';

  static Future<void> markLoggedIn() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_hasLoggedInKey, true);
  }

  static Future<bool> hasLoggedInBefore() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_hasLoggedInKey) ?? false;
  }

  static Future<void> clearSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_hasLoggedInKey);
  }

  static Future<void> refresh() async {
    await _storage.write(
      key: _key,
      value: DateTime.now().millisecondsSinceEpoch.toString(),
    );
  }
  
  static Future<void> forceExpire() async {
  await _storage.write(
    key: _key,
    value: DateTime.now()
        .subtract(const Duration(hours: 1))
        .millisecondsSinceEpoch
        .toString(),
  );
}

  static Future<bool> isExpired(int timeoutMinutes) async {
    final value = await _storage.read(key: _key);
    if (value == null) return true;

    final last =
        DateTime.fromMillisecondsSinceEpoch(int.parse(value));
    final diff = DateTime.now().difference(last);

    return diff.inMinutes >= timeoutMinutes;
  }

  static Future<void> clear() async {
    await _storage.delete(key: _key);
  }
}

class BiometricSettings {
  static const _key = 'biometrics_enabled';

  static Future<bool> isEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_key) ?? false;
  }

  static Future<void> setEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_key, value);
  }

  // static final _firestore = FirebaseFirestore.instance;
  // static final _auth = FirebaseAuth.instance;

  // static DocumentReference<Map<String, dynamic>> _doc() {
  //   final uid = _auth.currentUser!.uid;
  //   return _firestore.collection('users').doc(uid);
  // }

  // static Future<bool> isEnabled() async {
  //   final snap = await _doc().get();
  //   return snap.data()?['settings']?['biometricsEnabled'] ?? false;
  // }

  // static Future<void> setEnabled(bool value) async {
  //   await _doc().set({
  //     'settings': {
  //       'biometricsEnabled': value,
  //     }
  //   }, SetOptions(merge: true));
  // }

   // static Future<int> sessionTimeout() async {
   //   final snap = await _doc().get();
   //   return snap.data()?['settings']?['sessionTimeoutMinutes'] ?? 5;
   // }
}