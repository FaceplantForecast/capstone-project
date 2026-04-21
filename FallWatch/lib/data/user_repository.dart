import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class AppUser {
  final String uid;
  final String name;
  final String email;
  final String username;
  final String role;

  AppUser({
    required this.uid,
    required this.name,
    required this.email,
    required this.username,
    required this.role,
  });
  factory AppUser.fromFirestore(Map<String, dynamic> data, String uid) {
    return AppUser(
      uid: uid,
      name: data['name'],
      email: data['email'],
      username: data['username'],
      role: data['role'] ?? 'caretaker',
    );
  }
}

class UserRepository {
  static final _auth = FirebaseAuth.instance;
  static final _db = FirebaseFirestore.instance;

  static User? get firebaseUser => _auth.currentUser;

  static Future<AppUser?> getCurrentUserProfile() async {
    final user = firebaseUser;
    if (user == null) return null;

    final doc = 
      await _db.collection('users').doc(user.uid).get();

    if (!doc.exists) return null;

    return AppUser.fromFirestore(doc.data()!, user.uid);
  }

  static Future<void> logout() async {
    await _auth.signOut();
  }
}