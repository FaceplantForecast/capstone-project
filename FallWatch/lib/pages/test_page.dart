import 'package:flutter/material.dart';
import '../services/notification_services.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

Future<void> testFirestoreWrite() async {
  final db = FirebaseFirestore.instance;

  await db.collection('fall_events').add({
    'extra': 'EXTRAS',
    'fall_detected': true,
    'source': 'Test data',
    'timestamp': DateTime.now(),
    'created_at': FieldValue.serverTimestamp(),
  });

  print('✅ Test document written successfully!');
}

class TestPage extends StatefulWidget {
  const TestPage({super.key});

  @override
  State<TestPage> createState() => _TestPageState();
}
class _TestPageState extends State<TestPage> {

  void _showSnack(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildActionCard(
                icon: Icons.notification_important,
                title: "Test Notification",
                description: "Sends a manual fall alert.",
                color: Colors.orange,
                onTap: () {
                  NotificationService.showManualNotification(
                    title: 'Fall Detected',
                    body: 'A fall has been detected from your device.',
                  );
                  _showSnack("Notification Sent!");
                },
              ),

              const SizedBox(height: 20),

              _buildActionCard(
                icon: Icons.cloud_upload,
                title: "Test Firestore Write",
                description: "Writes a simulated fall event.",
                color: Colors.blue,
                onTap: () {
                  testFirestoreWrite();
                  _showSnack("Firestore Write Executed!");
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildActionCard({
    required IconData icon,
    required String title,
    required String description,
    required Color color,
    required VoidCallback onTap,
  }) {
    return SizedBox(
      width: 300,
      child: Card(
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(
              vertical: 20,
              horizontal: 16,
            ),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 24,
                  backgroundColor: color.withOpacity(0.15),
                  child: Icon(icon, size: 30, color: color),
                ),
                const SizedBox(width: 20),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(title,
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      Text(
                        description,
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}