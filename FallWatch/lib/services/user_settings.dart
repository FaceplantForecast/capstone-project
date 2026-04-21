import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class UserSettings {
  static final _firestore = FirebaseFirestore.instance;
  static final _auth = FirebaseAuth.instance;

  static Future<DocumentReference<Map<String, dynamic>>?> _doc() async {
    final user = _auth.currentUser;
    if (user == null) return null;
    return _firestore.collection('users').doc(user.uid);
  }

  static Future<Map<String, dynamic>> getSettings() async {
    final docRef = await _doc();

    if (docRef == null) return {};

    final snap = await docRef.get();

    if (!snap.exists) {
      // Create default settings doc if missing
      await docRef.set({
        'settings': {
          'biometricsEnabled': false,
          'sessionTimeoutMinutes': 5,
        }
      });
      return {
        'biometricsEnabled': false,
        'sessionTimeoutMinutes': 5,
      };
    }

    return snap.data()?['settings'] ?? {};
  }

  static Future<void> updateSetting(String key, dynamic value) async {
    final docRef = await _doc();
    if (docRef == null) return;

    await docRef.set({
      'settings.$key': value,
    }, SetOptions(merge: true));
  }
}