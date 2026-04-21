import 'package:cloud_firestore/cloud_firestore.dart';

class FallLogger {
  static final _dataBase = FirebaseFirestore.instance;

  static Future<void> logFallEvent(
    {
      required String source,
      required bool fallDetected,
      required DateTime timestamp,
      Map<String, dynamic>? extra,
    }
  )
  async {
    try {
      await _dataBase.collection('fall_events').add({
        'source' : source,
        'fall_detected' : fallDetected,
        'timestamp' : timestamp,
        'extra' : extra ?? {},
        'created_at' : FieldValue.serverTimestamp(),
      });
      print('✅ Fall event logged to Firestore');
    } catch (e) {
      print('⚠️ Failed to log fall event: $e');
    }
  }
}