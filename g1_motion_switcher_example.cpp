#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <string.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::robot::b2;

bool kbhit()
{
    static const int STDIN = 0;
    static bool initialized = false;

    if (!initialized) {
        termios term;
        tcgetattr(STDIN, &term);
        term.c_lflag &= ~ICANON;
        term.c_lflag &= ~ECHO;
        term.c_cc[VMIN] = 0;
        term.c_cc[VTIME] = 0;
        tcsetattr(STDIN, TCSANOW, &term);
        initialized = true;
    }

    int bytesWaiting;
    ioctl(STDIN, FIONREAD, &bytesWaiting);
    return bytesWaiting > 0;
}

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " network_interface" << std::endl;
        std::cout << "Example: " << argv[0] << " enp2s0" << std::endl;
        exit(0);
    }
    
    std::string networkInterface = argv[1];
    
    // 初始化通道工厂
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

    std::cout << "Initializing MotionSwitcherClient..." << std::endl;

    MotionSwitcherClient msc;
    msc.SetTimeout(5.0f);
    msc.Init();

    std::cout << "MotionSwitcherClient initialized successfully!" << std::endl;
    std::cout << "Press 'f' to check current motion mode" << std::endl;
    std::cout << "Press 'a' to select AI mode" << std::endl;
    std::cout << "Press 'r' to release motion control" << std::endl;
    std::cout << "Press 'q' to quit" << std::endl;

    char key = 0;
    while (true) {
        if (kbhit()) {
            key = getchar();
            
            switch(key) {
                case 'f':
                case 'F': {
                    std::string form, name;
                    int32_t ret = msc.CheckMode(form, name);
                    
                    if (ret == 0) {
                        std::cout << "CheckMode succeeded." << std::endl;
                        if (name.empty()) {
                            std::cout << "Current mode: No active motion control service" << std::endl;
                        } else {
                            std::cout << "Current mode: Service '" << name 
                                      << "' is active (form: " << form << ")" << std::endl;
                        }
                    } else {
                        std::cout << "CheckMode failed. Error code: " << ret << std::endl;
                    }
                    break;
                }
                
                case 'a':
                case 'A': {
                    std::cout << "Attempting to select AI mode..." << std::endl;
                    int32_t ret = msc.SelectMode("ai"); // 使用"ai"模式
                    
                    if (ret == 0) {
                        std::cout << "SelectMode 'ai' succeeded." << std::endl;
                    } else {
                        std::cout << "SelectMode 'ai' failed. Error code: " << ret << std::endl;
                    }
                    break;
                }
                
                case 'r':
                case 'R': {
                    std::cout << "Attempting to release motion control..." << std::endl;
                    int32_t ret = msc.ReleaseMode();
                    
                    if (ret == 0) {
                        std::cout << "ReleaseMode succeeded." << std::endl;
                    } else {
                        std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
                    }
                    break;
                }
                
                case 'q':
                case 'Q': {
                    std::cout << "Exiting program..." << std::endl;
                    return 0;
                }
                
                default:
                    break;
            }
        }
        
        usleep(10000); // 休眠10毫秒以避免过度占用CPU
    }

    return 0;
}