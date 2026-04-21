import 'package:local_auth/local_auth.dart';
import 'package:flutter/foundation.dart';

class BiometricService {
  static final _auth = LocalAuthentication();

  static Future<bool> authenticate() async {
    try {
      final supported =  await _auth.isDeviceSupported();
      final enrolled = await _auth.canCheckBiometrics;

      if (!supported || !enrolled) return false;

      return await _auth.authenticate(
        localizedReason: 'Unlock your account',
        );
    } catch (e) {
      debugPrint('Biometric error: $e');
      return false;
    }
  }
}