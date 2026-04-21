import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../pages/locked_page.dart';
import '../pages/user_page.dart';
import '../services/biometric_settings.dart';
import 'AppLockObserver.dart';

class AppLockGate extends StatefulWidget {
  const AppLockGate({super.key});

  @override
  State<AppLockGate> createState() => _AppLockGateState();
}

class _AppLockGateState extends State<AppLockGate> {
  bool _locked = false;
  bool _checking = false;
  String? _userNick;


  @override
  void initState() {
    super.initState();
    AppLockObserver.instance.needsLock.addListener(_handleLockChange);
    _initialize();
  }

  @override
  void dispose() {
    AppLockObserver.instance.needsLock.removeListener(_handleLockChange);
    super.dispose();
  }


  Future<void> _initialize() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final nick = await _getUserNickname(user);
    if (!mounted) return;

    setState(() => _userNick = nick);
  }

  Future<void> _handleLockChange() async {
    if (_checking) return;
    _checking = true;

    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      _checking = false;
      return;
    }

    final enabled = await BiometricSettings.isEnabled();
    final expired = await SecureSession.isExpired(5);

    if (!mounted) {
      _checking = false;
      return;
    }

    if (enabled &&
        (AppLockObserver.instance.needsLock.value || expired)) {

      // Schedule state change AFTER frame
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          setState(() => _locked = true);
        }
      });
    }

    _checking = false;
  }

  Future<String> _getUserNickname(User user) async {
    final doc = await FirebaseFirestore.instance
        .collection('users')
        .doc(user.uid)
        .get();
    

    return doc.data()?['username'] ??
        user.email!.split('@')[0];
  }

  void _unlock() {
    AppLockObserver.instance.clearLock();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() => _locked = false);
      }
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        UserPage(userNick: _userNick ?? 'User'),

        if (_locked)
          Positioned.fill(
            child: LockedPage(
              onUnlocked: _unlock,
            ),
          ),
      ],
    );
  }
}