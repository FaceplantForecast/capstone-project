import 'package:flutter/widgets.dart';

class AppLockObserver extends WidgetsBindingObserver {
  static final AppLockObserver instance = AppLockObserver._internal();
  AppLockObserver._internal();

  final ValueNotifier<bool> needsLock = ValueNotifier(false);

  void register() {
    WidgetsBinding.instance.addObserver(this);
  }

  void unregister() {
    WidgetsBinding.instance.removeObserver(this);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      needsLock.value = true;
    }
  }

  void clearLock() {
    needsLock.value = false;
  }
}