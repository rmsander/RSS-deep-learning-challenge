# 6.141 Deep Learning Challenge

This challenge was completed collaboratively as a part of MIT's Robotics: Science and Systems course.  The goal of this challenge was to design a vision and control system for autonomous navigation in a miniaturized urban environment.

This codebase was implemented and loaded onto our RACECAR platform, which operates using a Linux-based computer.  From here, we executed the following commands to prepare our RACECAR for autonomous navigation.

## Commands

For convenience, add this to your VM's `~/.bashrc`:
```bash
alias sc='sshpass -p racecar@mit ssh 192.168.1.46' # sudo apt-get install sshpass
alias epc='export ROS_MASTER_URI=http://racecar:11311; export ROS_IP=`hostname -I`'
```

6 Terminals (3. is in VM. The rest are on the car. Can't run 2. without 1.)
1. `sc` --> `teleop`
2. `sc` --> `rzz` (launches the camera)
3. `epc; rqt_image_view` (The original topic is `/zed/rgb/image_rect_color/compressed`. The visualized topic is `/labeled_image/compressed`.)
4. `sc` --> `rdo` (YOLO)
5. `sc` --> `rdv` (vision)
6. `sc` --> `rdc` (controller)

Everytime 4 or 5 restarts, `rqt_image_view` probably needs refreshing (click the refresh button). 

Because YOLO rarely works, item 4 might need to be run a few times. 5 and 6 can be run separately. If 4 works, try to not kill it. You can kill 5 and 6 to edit the code and run each of them again. 4 might break even 1 and 2 but those are easy to restart.

To go to the `deep_learning` directory, run `cdeep`.

For reference, the following aliases are already on the racecar so you can just run them.
```bash
alias rdo='roslaunch zed_wrapper zed.launch'
alias rdo='roslaunch deep_learning obj.launch'
alias rdv='roslaunch deep_learning vision.launch'
alias rdc='roslaunch deep_learning controller.launch'
alias cdeep='cd ~/racecar_ws/src/deep_learning'
```

## Other notes
- The TA car mode is being hardcoded. If needed, set `self.TA = True` in `controller.py`.
- If you want to record a rosbag for video, run `rosbag record /labeled_image/compressed`. Play it by `rosbag play your-bag-file.bag`. There's some delay when subscribing to the compressed topic, so the vision and obj nodes subscribe to the non-compressed one instead. It's probably best to just live screen recording if possible.
- Fix the repo? Somehow the `.git` directory in `deep_learning` gets corrupted. You can't just delete the current folder because there are YOLO model weights (ignored by `.gitignore`). Maybe search in google/stackoverflow. Git clone new one, then replace the new's `.git` with the old corrupt `.git`?

## Credits and Acknowledgements 
This project would not have been possible without my wonderful team members: James Pruegsanusanak, Jordan Docter, Mahi Elango, and Bhavik Nagda.  I would also like to thank the Robotics: Science and Systems staff for their support, mentorship, and guidance throughout this project.
