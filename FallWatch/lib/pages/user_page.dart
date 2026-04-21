import 'package:flutter/material.dart';
import 'dart:io' show Platform;
import 'package:firebase_messaging/firebase_messaging.dart';
import 'fall_history_page.dart';
import 'settings_page.dart';
import 'home_page.dart';
import 'test_page.dart';
import 'device_page.dart';
import 'graph_page.dart';
import '../data/user_repository.dart';
import '../services/device_connection.dart';
//import 'package:firebase_core/firebase_core.dart';

class UserPage extends StatefulWidget {
  final String userNick;
  const UserPage({super.key, required this.userNick});

  @override
  State<UserPage> createState() => _UserPageState();
}

class _UserPageState extends State<UserPage> {
  int _selectedIndex = 0;

  static const int _idxHome    = 0;
  static const int _idxTest    = 1;
  static const int _idxDevice  = 2;
  static const int _idxHistory = 3;
  static const int _idxGraph   = 4;
  static const int _idxSettings = 5;

  late final List<Widget> _pages;
  
  static const List<String> titles = [
    "Dashboard",
    'Test',
    'Device',
    'History',
    'Graph',
    'Settings',
  ];

  void _onItemTapped(int index) {
    setState(() => _selectedIndex = index);
  }
  
  Future<void> _logout() async {
  await UserRepository.logout();
  // Navigate back to login page
  Navigator.of(context).pushNamedAndRemoveUntil('/login', (route) => false);
}
  
  String? _fcmToken; // State variable to hold the FCM token

  @override
  void initState() {
    super.initState();
    DeviceMonitorService().start();
    _pages = [
      HomePage(
        title: 'Welcome',
        onNavigateToDevice: () => _onItemTapped(_idxDevice),
        onNavigateToHistory: () => _onItemTapped(_idxHistory),
        onNavigateToGraph: () => _onItemTapped(_idxGraph),
      ),
      const TestPage(),
      DevicePage(),
      const FallHistoryPage(),
      const GraphPage(),
      const SettingsPage(),
   ];

    if (!Platform.isWindows) {
      setupFCM();
    } else {
      print('Skipping FCM Notifs setup on Windows');
    } // Call FCM setup when the UserPage is initialized (not Windows)
  }

  void setupFCM() async {
    NotificationSettings settings = await FirebaseMessaging.instance.requestPermission();
    print('User granted permission: ${settings.authorizationStatus}');

    if (settings.authorizationStatus == AuthorizationStatus.authorized) {
      String? token = await FirebaseMessaging.instance.getToken();
    setState(() {
      _fcmToken = token;
    });
    print('FCM token: $_fcmToken');
    } else {
      print('User declined notification permissions');
    }
    

    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      print('FCM Foreground Message Received: ${message.notification?.title}');
    });
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.grey[350],
        foregroundColor: Theme.of(context).colorScheme.primary,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              titles[_selectedIndex],
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            Text(
              "Admin: ${widget.userNick}",
              style: const TextStyle(fontSize: 13),
            ),
          ],
        ),
        actions: [
          StreamBuilder(
            stream: DeviceMonitorService().stateStream,
            builder: (context, _) {
              final connected = DeviceMonitorService().connected;

              return Padding(
                padding: const EdgeInsets.only(right: 12),
                child: Row(
                  children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      width: 10,
                      height: 10,
                      decoration: BoxDecoration(
                        color: connected ? Colors.green : Colors.red,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      connected ? "Online" : "Offline",
                      style: const TextStyle(fontSize: 12),
                    ),
                  ],
                ),
              );
            },
          ),

          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: _logout,
          ),
        ],
      ),

      body: IndexedStack(
        index: _selectedIndex,
        children: _pages,
      ),

      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Theme.of(context).colorScheme.primary,
        unselectedItemColor: Colors.grey,
        elevation: 12,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined), label: "Home"),
          BottomNavigationBarItem(
            icon: Icon(Icons.notifications_none), label: "Test"),
          BottomNavigationBarItem(
            icon: Icon(Icons.devices_outlined), label: "Device"),
          BottomNavigationBarItem(
            icon: Icon(Icons.history), label: "History"),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart_outlined), label: "Graph"),  
          BottomNavigationBarItem(
            icon: Icon(Icons.settings_outlined), label: "Settings"),
        ],
      ),
    );
  }
}
//  @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: Text('Welcome, ${widget.userNick}'),
//       ),
//       drawer: Drawer(
//         child: ListView(
//           padding: EdgeInsets.zero,
//           children: [
//             const DrawerHeader(
//               decoration: BoxDecoration(
//                 color: Colors.blueAccent,
//               ),
//               child: Text('Nav Menu'),
//             ),
//             ListTile(
//               title: const Text('Home'),
//               leading: const Icon(Icons.home),
//               selected: _selectedIndex == 0,
//               onTap: () {
//                 _drawerNav(0);
//               },
//             ),
//             ListTile(
//               title: const Text('Test Notifications'),
//               leading: const Icon(Icons.hourglass_top_outlined),
//               selected: _selectedIndex == 1,
//               onTap: () {
//                 _drawerNav(1);
//               },
//             ),
//             ListTile(
//               title: const Text('Device'),
//               leading: const Icon(Icons.devices),
//               selected: _selectedIndex == 2,
//               onTap: () {
//                 _drawerNav(2);
//               },
//             ),
//             ListTile(
//               title: const Text('Fall History'),
//               leading: const Icon(Icons.history),
//               selected: _selectedIndex == 3,
//               onTap: () {
//                 _drawerNav(3);
//               },
//             ),
//             ListTile(
//               title: const Text('Settings'),
//               leading: const Icon(Icons.settings),
//               selected: _selectedIndex == 4,
//               onTap: () {
//                 _drawerNav(4);
//               },
//             ),
//             ListTile(
//               title: const Text('Logout'),
//               leading: const Icon(Icons.logout),
//               onTap: () async => await _logout(),
//             ),
//             Padding(
//               padding: const EdgeInsets.all(16.0),
//               child: SelectableText(
//                 'Token: ${_fcmToken ?? 'Loading...'}',
//                 style: const TextStyle(fontSize: 10, color: Colors.grey),
//               ),
//             ),
//           ]
//         )
//       ),
//       body: _widgetOptions.elementAt(_selectedIndex),
//     );
//   }
// }