import 'dart:async';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../services/device_connection.dart';
import '../services/notification_services.dart';
import 'test_page.dart';
import 'package:intl/intl.dart';

class HomePage extends StatefulWidget {
  final String title;
  final VoidCallback? onNavigateToDevice;
  final VoidCallback? onNavigateToHistory;
  final VoidCallback? onNavigateToGraph;

  const HomePage({
    super.key, 
    required this.title,
    this.onNavigateToDevice,
    this.onNavigateToHistory,
    this.onNavigateToGraph,
  });

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;
  Timer? _stopTimer;

  late Stream<QuerySnapshot> fallEventsStream;
  final DeviceMonitorService service = DeviceMonitorService();

  bool _isLiveAlert = false;
  DateTime? _lastProcessedFallTime;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    fallEventsStream = FirebaseFirestore.instance
      .collection('fall_events')
      .orderBy('created_at', descending: true)
      .limit(5)
      .snapshots();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );

    _pulseAnimation =
      Tween<double>(begin: 1.0, end: 1.08).animate(
    CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ),
    );
    _resumeAlertIfNeeded();
  }



  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _pulseController.dispose();
    _stopTimer?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _resumeAlertIfNeeded();
    }
  }

  void _resumeAlertIfNeeded() {
    if (service.fallDetected && !_isLiveAlert) {
      _lastProcessedFallTime = service.lastTimestamp;
      WidgetsBinding.instance.addPostFrameCallback((_) => _triggerLiveAlert()); 
    }
  }

  String _formatTimestamp(dynamic ts) {
    if (ts == null) return 'Unknown';
    DateTime dt;
    if (ts is Timestamp) { dt = ts.toDate(); }
    else if (ts is DateTime) { dt = ts; }
    else { dt = DateTime.tryParse(ts.toString()) ?? DateTime.now(); }
    return DateFormat('MMM d, yyyy • HH:mm').format(dt);
  }

  void _triggerLiveAlert() {
    if (!mounted) return;
    setState(() => _isLiveAlert = true);
    _pulseController.repeat(reverse: true);
    _stopTimer?.cancel();
    _stopTimer = Timer(const Duration(seconds: 15), () {
      if (mounted) {
        _pulseController..stop()..reset();
      }
    });
  }

  void _acknowledgeAlert() {
    service.acknowledgeFall();
    _pulseController..stop()..reset();
    _stopTimer?.cancel();
    setState(() => _isLiveAlert = false);
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(    
        child: StreamBuilder<void>(
          stream: service.stateStream,
          builder: (context, _) {
            final serviceFall = service.fallDetected;
            final fallTime = service.lastTimestamp;

            if (serviceFall && fallTime != null) {
              if (_lastProcessedFallTime == null || fallTime.isAfter(_lastProcessedFallTime!)) {
              _lastProcessedFallTime = fallTime;
              WidgetsBinding.instance
                  .addPostFrameCallback((_) => _triggerLiveAlert());
              }
            }

            if (!serviceFall && _isLiveAlert) {
              WidgetsBinding.instance.addPostFrameCallback((_) {
                if (mounted) {
                  _pulseController..stop()..reset();
                  _stopTimer?.cancel();
                  setState(() => _isLiveAlert = false);
                  _acknowledgeAlert();
                }
              });
            }

            final rawDeviceId = service.fallSourceDevice;
            final locationLabel = rawDeviceId != null
              ?(service.devices[rawDeviceId]?.label ?? rawDeviceId)
              : null;

            return StreamBuilder<QuerySnapshot>(
            // Firestore stream powers the Recent Activity list.
            stream: fallEventsStream,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const Center(child: CircularProgressIndicator());
              }

              final docs = snapshot.data?.docs ?? [];

              return SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildAlertCard(locationLabel),
                    // Recent Activity

                    const SizedBox(height: 32),

            // recents
                   const Text(
                      "Recent Activity",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 16),

                    if (docs.isEmpty)
                      const Text("No recent events.")
                    else
                      ...docs.map((doc) {
                        final data = doc.data() as Map<String, dynamic>;
                        final fall = data['fall_detected'] ?? false;
                        final src = data['source'] ?? 'Unknown';
                        final time = _formatTimestamp(data['timestamp']);

                        return Container(
                          margin: const EdgeInsets.only(bottom: 12),
                          padding: const EdgeInsets.all(18),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(18),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(.04),
                                blurRadius: 10,
                                offset: const Offset(0, 4),
                              )
                            ],
                          ),
                          child: Row(
                            children: [
                              Icon(
                                fall ? Icons.error : Icons.check_circle,
                                color: fall ? Colors.red : Colors.green,
                              ),
                              const SizedBox(width: 14),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      fall ? "Fall Detected" : "Normal Activity",
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w600),
                                    ),
                                    Text(
                                      "$src • $time",
                                      style: const TextStyle(
                                        fontSize: 13, color: Colors.grey),
                                    ),
                                  ],
                                ),
                              )
                            ],
                          ),
                        );
                      }),

                    const SizedBox(height: 32),

                    /// quick actions
                    const Text(
                      "Quick Actions",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 16),

                    GridView.count(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      crossAxisCount: 2,
                      mainAxisSpacing: 14,
                      crossAxisSpacing: 14,
                      childAspectRatio: 3.2,
                      children: [
                        _ActionCard(
                          icon: Icons.history,
                          label: "History",
                          onTap: () => 
                            widget.onNavigateToHistory?.call(),
                        ),
                        _ActionCard(
                          icon: Icons.devices,
                          label: "Device",
                          onTap: () =>
                            widget.onNavigateToDevice?.call(),
                        ),
                        _ActionCard(
                          icon: Icons.notifications,
                          label: "Test Alert",
                          onTap: () {
                            service.fallDetected = true;
                            service.fallSourceDevice = 'sim-pi-01';
                            service.lastTimestamp = DateTime.now();
                            service.stateStream;
                            _stateController_notify();
                            NotificationService.showManualNotification(
                              title: 'Fall Detected',
                              body: 'Test notification',
                            );
                            testFirestoreWrite();
                          },
                        ),
                        // {"type":"data","payload":{"type":"status","event":"power_state","status":"System on Battery Backup","device":"raspberry_pi","timestamp":1772127905.0061448}}
                        // {"type":"data","payload":{"type":"status","event":"power_state","status":"System on Wall Power","device":"raspberry_pi","timestamp":1772127879.9227018}}
                        _ActionCard(
                          icon: Icons.bar_chart,
                          label: "Graph",
                          onTap: () => 
                            widget.onNavigateToGraph?.call(),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            },
          );
        },
      ),
    );
  }

  void _stateController_notify() {
    // forces an immediate rebuild by rebuilding setState
    setState(() {});
  }

  Widget _buildAlertCard(String? locationLabel) {
    final showRed = _isLiveAlert || service.fallDetected;
    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) => Transform.scale(
        scale: _isLiveAlert ? _pulseAnimation.value : 1.0,
        child: child,
      ),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          gradient: LinearGradient(
            colors: _isLiveAlert
                ? [Colors.red.shade400, Colors.red.shade600]
                : [Colors.green.shade400, Colors.green.shade600],
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(
              _isLiveAlert
                  ? Icons.warning_rounded
                  : Icons.shield_outlined,
              color: Colors.white,
              size: 42,
            ),
            const SizedBox(height: 12),
            Text(
              _isLiveAlert ? "Fall Detected" : "Monitoring Active",
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 6),
 
            // Location line — only shown when a fall is active.
            if (showRed && locationLabel != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  children: [
                    const Icon(Icons.location_on,
                        color: Colors.white70, size: 16),
                    const SizedBox(width: 4),
                    Text(
                      locationLabel,
                      style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
 
            Text(
              _isLiveAlert
                  ? service.lastTimestamp != null
                      ? "Detected at ${_formatTimestamp(service.lastTimestamp)}"
                      : "Fall event received"
                  : "No recent fall events detected.",
              style: const TextStyle(color: Colors.white70),
            ),
 
            // Acknowledge button.
            if (showRed) ...[
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: _acknowledgeAlert,
                icon: const Icon(Icons.check),
                label: const Text("Acknowledge"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.red.shade700,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionCard({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(20),
      onTap: onTap,
      child: Ink(
        decoration: BoxDecoration(
          color: const Color.fromARGB(255, 166, 205, 212),
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(.05),
              blurRadius: 12,
              offset: const Offset(0, 6),
            )
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 24),
            const SizedBox(height: 8),
            Flexible(
              child: Text(
                label,
                textAlign: TextAlign.center,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            )
          ],
        ),
      ),
    );
  }
}
