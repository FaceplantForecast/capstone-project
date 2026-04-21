import 'package:flutter/material.dart';

class LogoHeader extends StatelessWidget {
  final String title;
  const LogoHeader({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const CircleAvatar(
          radius: 25,
          backgroundImage: AssetImage('assets/fallwatch_nb.png'),
        ),
        const SizedBox(width: 8),
        Text(title),
      ]
    );
  }
}