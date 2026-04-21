import 'package:flutter/material.dart';

class WelcomeWidget extends StatelessWidget {
  final String title;
  const WelcomeWidget({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text(
        'Welcome, $title!',
        style: const TextStyle(fontSize: 24),
      ),
    );
  }
}