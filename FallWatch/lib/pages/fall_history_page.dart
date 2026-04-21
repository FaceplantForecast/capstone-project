import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class FallHistoryPage extends StatelessWidget {
  const FallHistoryPage({super.key});

  String _formatTimestamp(dynamic ts) {
    DateTime dt;

    if (ts is Timestamp) {
      dt = ts.toDate();
    } else if (ts is DateTime) {
      dt = ts;
    } else if (ts is String) {
      dt = DateTime.tryParse(ts) ?? DateTime.now();
    } else {
      dt = DateTime.now();
    }

    return '${dt.month}/${dt.day}/${dt.year} ${dt.hour}:${dt.minute.toString().padLeft(2,'0')}';
  }

  @override
  Widget build(BuildContext context) {
    final fallEventsStream = FirebaseFirestore.instance
      .collection('fall_events')
      .orderBy('created_at', descending: true)
      .snapshots();
    return Scaffold(
      body: StreamBuilder<QuerySnapshot>(
        stream: fallEventsStream,      
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
            return const Center(child: Text('No fall events recorded.'));
          }
          final docs = snapshot.data!.docs;

          return ListView.builder(
            padding: const EdgeInsets.all(12),
            itemCount: docs.length,
            itemBuilder: (context, index) {
              final data = docs[index].data()! as Map<String, dynamic>;
              final fallDetected = data['fall_detected'] ?? false;
              final source = data['source'] ?? 'Unknown';
              final timestamp = _formatTimestamp(data['timestamp']);
              final extra = data['extra'];

              return Card(
                elevation: 3,
                margin: const EdgeInsets.symmetric(vertical: 8),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Status Icon
                      CircleAvatar(
                        radius: 22,
                        backgroundColor:
                            fallDetected ? Colors.red.shade100 : Colors.green.shade100,
                        child: Icon(
                          fallDetected
                              ? Icons.warning_amber_rounded
                              : Icons.check_circle,
                          color: fallDetected ? Colors.red : Colors.green,
                          size: 26,
                        ),
                      ),

                      const SizedBox(width: 12),

                      // Event Info
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // Title + Time
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  fallDetected ? 'Fall Detected' : 'No Fall',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color:
                                        fallDetected ? Colors.red : Colors.green,
                                  ),
                                ),
                                Text(
                                  timestamp,
                                  style: const TextStyle(
                                    fontSize: 12,
                                    color: Colors.grey,
                                  ),
                                ),
                              ],
                            ),

                            const SizedBox(height: 6),

                            // Source
                            Row(
                              children: [
                                const Icon(Icons.sensors, size: 16, color: Colors.grey),
                                const SizedBox(width: 6),
                                Text(
                                  'Source: $source',
                                  style: const TextStyle(fontSize: 13),
                                ),
                              ],
                            ),

                            // Extra Data (optional)
                            if (extra != null && extra.toString().isNotEmpty) ...[
                              const SizedBox(height: 6),
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Icon(Icons.info_outline,
                                      size: 16, color: Colors.grey),
                                  const SizedBox(width: 6),
                                  Expanded(
                                    child: Text(
                                      extra.toString(),
                                      style: const TextStyle(
                                        fontSize: 12,
                                        color: Colors.black87,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          );
        },
      ),
      
    );
  }
}