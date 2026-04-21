import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';

class GraphPage extends StatefulWidget {
  const GraphPage({super.key});

  @override
  State<GraphPage> createState() => _GraphPageState();
}

class _GraphPageState extends State<GraphPage> {
  late Stream<QuerySnapshot> fallEventsStream;

  String _selectedRange = '7d';
  DateTimeRange? _dateRange;

  @override
  void initState() {
    super.initState();
    _updateStream();
  }

  void _updateStream() {
    final now = DateTime.now();
    DateTime end = now;
    DateTime start;

    switch (_selectedRange) {
      case '7d':
        start = end.subtract(const Duration(days: 7));
        break;
      case '30d':
        start = end.subtract(const Duration(days: 30));
        break;
      case 'custom':
        if (_dateRange != null) {
          start = _dateRange!.start;
          end = _dateRange!.end;
        } else {
          start = end.subtract(const Duration(days: 7));
        }
        break;
      default:
        start = end.subtract(const Duration(days: 7));;
    }

    fallEventsStream = FirebaseFirestore.instance
        .collection('fall_events')
        .where('timestamp', isGreaterThanOrEqualTo: Timestamp.fromDate(start))
        .where('timestamp', isLessThanOrEqualTo: Timestamp.fromDate(now))
        .orderBy('timestamp', descending: false)
        .snapshots();
    setState(() {});
  }

  List<String> _generateDateRange(DateTime start, DateTime end) {
  List<String> days = [];
  DateTime current = DateTime(start.year, start.month, start.day);

  while (!current.isAfter(end)) {
    days.add(DateFormat('yyyy-MM-dd').format(current));
    current = current.add(const Duration(days: 1));
  }

  return days;
}

  Map<String, int> _aggregateFalls(List<QueryDocumentSnapshot> docs, DateTime start, DateTime end) {
    Map<String, int> fallCountByDay = {
      for (var day in _generateDateRange(start, end)) day: 0
    };

    for (var doc in docs) {
      final data = doc.data() as Map<String, dynamic>;
      final fallDetected = data['fall_detected'] ?? false;
      final ts = data['timestamp'];

      if (fallDetected && ts != null) {
        DateTime date;
        if (ts is Timestamp) {
          date = ts.toDate();
        } else if (ts is DateTime) {
          date = ts;
        } else {
          date = DateTime.tryParse(ts.toString()) ?? DateTime.now();
        }

        final day = DateFormat('yyyy-MM-dd').format(date);
        if (fallCountByDay.containsKey(day)) {
          fallCountByDay[day] = (fallCountByDay[day] ?? 0) + 1;
        }
      }
    }

    return fallCountByDay;
  }

  Future<void> _selectCustomRange(BuildContext context) async {
    final now = DateTime.now();
    final lastMonth = now.subtract(const Duration(days: 30));

    final picked = await showDateRangePicker(
      context: context,
      firstDate: DateTime(2025),
      lastDate: now,
      initialDateRange:
          _dateRange ?? DateTimeRange(start: lastMonth, end: now),
    );

    if (picked != null) {
      setState(() {
        _selectedRange = 'custom';
        _dateRange = picked;
      });
      _updateStream();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          const SizedBox(height: 12),
          _buildRangeSelector(context),
          const Divider(thickness: 1),
          Expanded(
            child: StreamBuilder<QuerySnapshot>(
              stream: fallEventsStream,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }

                if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                  return const Center(child: Text('No fall events recorded.'));
                }
                final now = DateTime.now();
                DateTime end = now;
                DateTime start;


                switch (_selectedRange) {
                  case '7d':
                    start = end.subtract(const Duration(days: 7));
                    break;
                  case '30d':
                    start = end.subtract(const Duration(days: 30));
                    break;
                  case 'custom':
                    start = _dateRange!.start;
                    end = _dateRange!.end;
                    break;
                  default:
                    start = end.subtract(const Duration(days: 7));
                }

                final fallData = _aggregateFalls(snapshot.data!.docs, start, end);

                if (fallData.isEmpty) {
                  return const Center(child: Text('No falls detected.'));
                }

                final maxYValue = fallData.values.isEmpty
                  ? 1
                  : fallData.values.reduce((a, b) => a > b ? a : b);

                final yInterval = (maxYValue / 5).ceil().toDouble();

                final sortedKeys = fallData.keys.toList()..sort();
                final spots = sortedKeys.asMap().entries.map((entry) {
                  final index = entry.key.toDouble();
                  final day = entry.value;
                  return FlSpot(index, fallData[day]!.toDouble());
                }).toList();

                return Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Text(
                        'Fall Frequency (${_getRangeLabel()})',
                        style: Theme.of(context)
                            .textTheme
                            .headlineSmall
                            ?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: Colors.blueAccent,
                            ),
                      ),
                      const SizedBox(height: 20),
                      Expanded(
                        child: Row(
                          children: [
                            // y-axis name
                            RotatedBox(
                              quarterTurns: 3,
                              child: Text(
                                'Number of Falls',
                                style: TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.grey[700],
                                ),
                              ),
                            ),

                            const SizedBox(width: 8),

                            // Chart and x-axis name
                            Expanded(
                              child: Column(
                                children: [
                                  Expanded(
                                    child: LineChart(
                                      LineChartData(
                                        lineTouchData: LineTouchData(
                                          enabled: true,
                                          touchTooltipData: LineTouchTooltipData(
                                            getTooltipItems: (touchedSpots) {
                                              return touchedSpots.map((spot) {
                                                return LineTooltipItem(
                                                  'Falls: ${spot.y.toInt()}',
                                                  const TextStyle(
                                                    color: Colors.white,
                                                    fontWeight: FontWeight.bold,
                                                  ),
                                                );
                                              }).toList();
                                            },
                                          ),
                                        ),
                                        
                                        minY: 0,
                                        minX: spots.isEmpty ? 0 : spots.first.x,
                                        maxX: spots.isEmpty ? 1 : spots.last.x,
                                        maxY: (fallData.values.isEmpty
                                                ? 1
                                                : fallData.values.reduce((a, b) => a > b ? a : b))
                                            .toDouble() +
                                            1,
                                        gridData: FlGridData(
                                          show: true,
                                          horizontalInterval: 1,
                                          getDrawingHorizontalLine: (value) =>
                                              FlLine(strokeWidth: 0.5, color: Colors.grey),
                                          getDrawingVerticalLine: (value) =>
                                              FlLine(strokeWidth: 0.3, color: Colors.grey.shade300),
                                        ),
                                        borderData: FlBorderData(show: false),
                                        titlesData: FlTitlesData(
                                          topTitles:
                                              AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                          rightTitles:
                                              AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                          bottomTitles: AxisTitles(
                                            sideTitles: SideTitles(
                                              showTitles: true,
                                              interval: _selectedRange == '30d' ? 5 : 1,
                                              reservedSize: 30,
                                              getTitlesWidget: (value, meta) {
                                                final index = value.toInt();
                                                if (index < 0 || index >= sortedKeys.length) {
                                                  return const SizedBox();
                                                }
                                                final dateStr = sortedKeys[index];
                                                final formatted =
                                                    DateFormat('MM/dd').format(DateTime.parse(dateStr));
                                                return Text(
                                                  formatted,
                                                  style: const TextStyle(fontSize: 10),
                                                );
                                              },
                                            ),
                                          ),
                                          leftTitles: AxisTitles(
                                            sideTitles: SideTitles(
                                              showTitles: true,
                                              interval: yInterval,
                                              reservedSize: 36,
                                              getTitlesWidget: (value, meta) => Text(
                                                value.toInt().toString(),
                                                style: const TextStyle(fontSize: 12),
                                              ),
                                            ),
                                          ),
                                        ),
                                        lineBarsData: [
                                          LineChartBarData(
                                            spots: spots,
                                            isCurved: false,
                                            color: Colors.redAccent,
                                            barWidth: 3,
                                            dotData: FlDotData(show: true),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),

                                  const SizedBox(height: 6),

                                  // X-axis label
                                  Text(
                                    'Date',
                                    style: TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.w600,
                                      color: Colors.grey[700],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  String _getRangeLabel() {
    switch (_selectedRange) {
      case '7d':
        return 'Last 7 Days';
      case '30d':
        return 'Last 30 Days';
      case 'custom':
        if (_dateRange != null) {
          return '${DateFormat('MM/dd').format(_dateRange!.start)} - ${DateFormat('MM/dd').format(_dateRange!.end)}';
        }
        return 'Custom Range';
      default:
        return 'Last 7 Days';
    }
  }

  Widget _buildRangeSelector(BuildContext context) {
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 8,
      children: [
        _rangeButton('7d', '7 Days'),
        _rangeButton('30d', '30 Days'),
        ElevatedButton.icon(
          onPressed: () => _selectCustomRange(context),
          icon: const Icon(Icons.date_range),
          label: const Text('Custom'),
          style: ElevatedButton.styleFrom(
            backgroundColor: _selectedRange == 'custom'
                ? Colors.blueAccent
                : Colors.grey[400],
          ),
        ),
      ],
    );
  }

  Widget _rangeButton(String value, String label) {
    final bool isSelected = _selectedRange == value;
    return ElevatedButton(
      onPressed: () {
        setState(() => _selectedRange = value);
        _updateStream();
      },
      style: ElevatedButton.styleFrom(
        backgroundColor: isSelected ? Colors.blueAccent : Colors.grey[400],
      ),
      child: Text(label),
    );
  }
}