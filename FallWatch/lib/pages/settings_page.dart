import 'package:flutter/material.dart';
import '../services/user_settings.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../services/biometric_settings.dart';
import '../services/secure_credentials.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class PasswordConfirmDialog {
  static Future<String?> show(BuildContext context) async {
    final controller = TextEditingController();
    bool loading = false;
    String? error;

    return await showDialog<String?>(
      context: context,
      barrierDismissible: false,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('Confirm Password'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    'Enter your password to enable biometric login.',
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: controller,
                    obscureText: true,
                    decoration: InputDecoration(
                      labelText: 'Password',
                      errorText: error,
                    ),
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed:
                      loading ? null : () => Navigator.pop(context, null),
                  child: const Text('Cancel'),
                ),
                ElevatedButton(
                  onPressed: loading
                      ? null
                      : () async {
                          setDialogState(() {
                            loading = true;
                            error = null;
                          });

                          try {
                            final user = FirebaseAuth.instance.currentUser;

                            if (user == null || user.email == null) {
                              Navigator.pop(context, null);
                              return;
                            }

                            await FirebaseAuth.instance.signInWithEmailAndPassword(
                              email: user.email!,
                              password: controller.text.trim(),
                            );

                            Navigator.pop(
                                context, controller.text.trim());
                          } on FirebaseAuthException catch (e) {
                            setDialogState(() {
                              error = e.code == 'wrong-password'
                                  ? 'Incorrect password'
                                  : e.message ?? 'Authentication failed';
                              loading = false;
                            });
                          } catch (_) {
                            setDialogState(() {
                              error = 'Unexpected error';
                              loading = false;
                            });
                          }
                        },
                  child: loading
                      ? const SizedBox(
                          height: 18,
                          width: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                          ),
                        )
                      : const Text('Confirm'),
                ),
              ],
            );
          },
        );
      },
    );
  }
}
class _SettingsPageState extends State<SettingsPage> {
  bool biometricsEnabled = false;
  int sessionTimeout = 5;
  bool loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final enabled = await BiometricSettings.isEnabled();

    if (!mounted) return;

    setState(() {
      biometricsEnabled = enabled;
      loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (loading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: ListView(
        children: [
          const ListTile(title: Text('Security')),
          SwitchListTile(
            title: const Text('Enable biometric login'),
            value: biometricsEnabled,
            onChanged: (v) async {
              if (v) {
                final password = await PasswordConfirmDialog.show(context);
                if (password == null) return;

                final user = FirebaseAuth.instance.currentUser;
                if (user == null || user.email == null) return;

                // Save credentials securely
                await SecureCredentials.save(user.email!, password);
              } else {
                await SecureCredentials.clear();
              }

              await BiometricSettings.setEnabled(v);

              if (!mounted) return;

              setState(() {
                biometricsEnabled = v;
              });
            },
          ),
          ListTile(
            title: const Text('Session timeout'),
            subtitle: Text('$sessionTimeout minutes'),
            trailing: DropdownButton<int>(
              value: sessionTimeout,
              items: const [
                DropdownMenuItem(value: 1, child: Text('1 min')),
                DropdownMenuItem(value: 5, child: Text('5 min')),
                DropdownMenuItem(value: 15, child: Text('15 min')),
              ],
              onChanged: (v) async {
                if (v == null) return;
                setState(() => sessionTimeout = v);
                await UserSettings.updateSetting(
                  'sessionTimeoutMinutes',
                  v,
                );
              },
            ),
          ),
          ListTile(
            title: const Text(
              'DEV: Force biometric re-auth',
              style: TextStyle(color: Colors.red),
            ),
            onTap: () async {
              await SecureSession.forceExpire();
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Session expired. Lock on resume.')),
              );
            },
          ),
        ],
      ),

    );
  }
}