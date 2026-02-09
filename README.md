# Faceplant Forecast Radar Human Activity Detector
This is the Github repository for the code running on the Raspberry Pi 5 in Team 20: Human Activity Detector 2 (Faceplant Forecast) for ECEN 403/404 at Texas A&M University. This project spans from Fall 2025 to Spring 2026.

# Group Members

 **Team Lead/Radar Subsystem:** Fritz Hesse<br>
 **AI and Power Subsystems:** Charles Marks<br>
 **App and Communication Subsystems:** Henry Moyes

## Operation

Will add more details later.
This project uses a TI AWR2944EVM radar module to gather data which is then processed by a custom AI model to detect falls and send out notifications.

## Notes
There are a few oddities in the code and repository as a whole, such as the lack of training data output from the collection script. That is to reduce file size of the repository. There are also multiple possible serial port names for connecting to the radar. This is because the Raspberry Pi sometimes changes the naming scheme of the USB ports upon boot, and we have yet to find the reason.
