import 'package:flutter/material.dart';
import '../services/biometric_service.dart';
import '../services/biometric_settings.dart';
// import '../widgets/AppLockObserver.dart';
// import 'user_page.dart';
// import 'login_page.dart';

class LockedPage extends StatefulWidget {
  final VoidCallback onUnlocked;

  const LockedPage({super.key, required this.onUnlocked});

  @override
  State<LockedPage> createState() => _LockedPageState();
}

class _LockedPageState extends State<LockedPage> {
  bool _authenticating = false;

  @override
  void initState() {
    super.initState();

    // Delay biometric call until frame completes
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _attemptUnlock();
    });
  }

  Future<void> _attemptUnlock() async {
    if (_authenticating) return;
    _authenticating = true;

    await Future.delayed(const Duration(milliseconds: 300));

    final success = await BiometricService.authenticate();

    if (!mounted) return;

    if (success) {
      await SecureSession.refresh();
      widget.onUnlocked();
    } else {
      _authenticating = false;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Authentication failed')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.black.withOpacity(0.6),
      child: Center(
        child: Card(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: const [
                Icon(Icons.lock, size: 48),
                SizedBox(height: 16),
                Text(
                  'Session Locked',
                  style: TextStyle(fontSize: 18),
                ),
                SizedBox(height: 20),
                CircularProgressIndicator(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}