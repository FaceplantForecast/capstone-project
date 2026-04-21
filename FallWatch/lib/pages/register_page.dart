import 'package:flutter/material.dart';
import '../widgets/LogoHeader.dart';
import '../widgets/Register.dart';

class RegisterPage extends StatefulWidget {
  final String title;
  const RegisterPage({super.key, required this.title});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: LogoHeader(title: 'Register'),
      ),
      body: Container(decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Color(0xFFE8F1FA),
            Colors.blueGrey,
          ],
        ),
      ),
      child: RegisterForm(),
    )  
    
    );
  }
}