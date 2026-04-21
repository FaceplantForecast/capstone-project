import 'package:flutter/material.dart';
import '../services/biometric_settings.dart';
import '../services/biometric_service.dart';
import '../services/secure_credentials.dart';
import '../pages/user_page.dart';
import '../pages/register_page.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';


class LoginForm extends StatefulWidget {
  const LoginForm({super.key});

  @override
  State<LoginForm> createState() => _LoginFormState();
}

class _LoginFormState extends State<LoginForm> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  bool _isLoading = false;

  bool _biometricsAvailable = false;

  @override
  void initState() {
    super.initState();
    _initBiometrics();
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }
  Future<void> _initBiometrics() async {
    try {
      final creds = await SecureCredentials.read();

      if (!mounted) return;

      setState(() {
        _biometricsAvailable =
            creds['email'] != null && creds['password'] != null;
      });
    } catch (e) {
      debugPrint("Biometric init error: $e");

      if (!mounted) return;

      setState(() {
        _biometricsAvailable = false;
      });
    }
  }

  Future<void> _unlockWithBiometrics() async {
    final success = await BiometricService.authenticate();
    if (!success || !mounted) return;

    final creds = await SecureCredentials.read();
    final email = creds['email'];
    final password = creds['password'];

    if (email == null || password == null) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('No saved credentials found')),
    );
    return;
  }
    // final user = FirebaseAuth.instance.currentUser;
    
    // if (user == null) {
    //   ScaffoldMessenger.of(context).showSnackBar(
    //     const SnackBar(
    //       content: Text('Please log in with email and password first'),
    //     ),
    //   );
    //   return;
    // }
    setState(() => _isLoading = true);
    try { 
        final credential =
            await FirebaseAuth.instance.signInWithEmailAndPassword(
          email: email.toString(),
          password: password.toString(),
        );

        final user = credential.user!;

        final userNick = await _getUserNickname(user);

        await SecureSession.refresh();

        if (!mounted) return;
        await SecureSession.markLoggedIn();

        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => UserPage(userNick: userNick),
          ),
        );
        return;
    } catch (_) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Biometric login failed')),
        );
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<String> _getUserNickname(User user) async {
    final doc = await FirebaseFirestore.instance
        .collection('users')
        .doc(user.uid)
        .get();

    return doc.data()?['username'] ??
        user.email!.split('@')[0];
  }

  Future<void> _loginUser() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final email = _emailController.text.trim();
      final password = _passwordController.text.trim();

      final credential = await FirebaseAuth.instance.signInWithEmailAndPassword(
        email : email,
        password : password,
      );

      final user = credential.user!;

      await SecureCredentials.save(email, password);

      if (!user.emailVerified) {
        await user.sendEmailVerification();
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: 
          Text('Please verify your email before logging in.'),
          ),
        );
        await FirebaseAuth.instance.signOut();
        return;
      }

      final userNick = await _getUserNickname(user);
      

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Welcome back, $userNick!')),
      );

      await SecureSession.refresh();

      if (!mounted) return;
      await SecureSession.markLoggedIn();
      
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => UserPage(userNick: userNick),
          ),
        );
    } on FirebaseAuthException catch(e) {
      String message = 'Login failed';

      if (e.code == 'user-not-found') {
        message = 'No account found for this email';
      } else if (e.code == 'wrong-password') {
        message = 'Incorrect password';
      } else if (e.code == 'invalid-email') {
        message = 'Invalid email address';
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }
    
  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 40),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 480),
          child: Card(
            elevation: 10,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(24),
            ),
            child: Padding(
              padding: const EdgeInsets.all(28),
              child: Form(
                key: _formKey,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [

                    Image.asset('assets/fallwatch_nb.png', width: 175, height: 175),
                    const SizedBox(height: 5),

                    const Text(
                      "Welcome to FallWatch",
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      "Sign in to continue",
                      style: TextStyle(color: Colors.grey),
                    ),

                    const SizedBox(height: 30),

                    TextFormField(
                      controller: _emailController,
                      decoration: InputDecoration(
                        prefixIcon: const Icon(Icons.email_outlined),
                        labelText: 'Email',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),

                    const SizedBox(height: 18),

                    TextFormField(
                      controller: _passwordController,
                      obscureText: true,
                      decoration: InputDecoration(
                        prefixIcon: const Icon(Icons.lock_outline),
                        labelText: 'Password',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),

                    const SizedBox(height: 26),

                    SizedBox(
                      width: double.infinity,
                      height: 52,
                      child: ElevatedButton(
                        onPressed: _isLoading ? null : _loginUser,
                        style: ElevatedButton.styleFrom(
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                        child: const Text("Login"),
                      ),
                    ),

                    const SizedBox(height: 14),

                    TextButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) =>
                                const RegisterPage(title: 'Register'),
                          ),
                        );
                      },
                      child: const Text("Create an account"),
                    ),

                    if (_biometricsAvailable) ...[
                      const SizedBox(height: 16),
                      const Divider(),
                      const SizedBox(height: 12),
                      OutlinedButton.icon(
                        icon: const Icon(Icons.fingerprint),
                        label: const Text("Unlock with Biometrics"),
                        onPressed: _unlockWithBiometrics,
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
