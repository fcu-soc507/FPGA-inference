#include <ros/ros.h>
#include <std_msgs/ByteMultiArray.h>

#include <string>
#include <vector>
#include <iostream>
#include <arpa/inet.h>

#include <sys/types.h>   
#include <sys/socket.h>   
#include <netinet/in.h>   
#include <unistd.h>   
#include <errno.h>   
#include <fcntl.h>

#define SERV_PORT   8000   

int udpserver_open(unsigned short port,const char* addr = "0.0.0.0") {
    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock_fd < 0) { return sock_fd;  }
 
    auto flags = fcntl(sock_fd, F_GETFL, 0);
    fcntl(sock_fd, F_SETFL, flags|O_NONBLOCK);

    struct sockaddr_in addr_serv;  
    int len;  
    memset(&addr_serv, 0, sizeof(struct sockaddr_in));
    addr_serv.sin_family = AF_INET;
    addr_serv.sin_port = htons(port);
    inet_pton(AF_INET, addr, &addr_serv.sin_addr);
    len = sizeof(addr_serv);  

    int optval = 1;
    setsockopt(sock_fd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval)); 

    auto res = bind(sock_fd, (struct sockaddr *)&addr_serv, sizeof(addr_serv));
    if(res < 0)  
    {
        close(sock_fd);
        perror("bind error:");  
        return res;
    }  

    return sock_fd;
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "bbox_publish");

    ros::NodeHandle n;
    int recv_num;  

    ros::Publisher publisher = n.advertise<std_msgs::ByteMultiArray>("bbox_publish", 1000);

    auto server_fd = udpserver_open(SERV_PORT);

    ros::Rate loop_rate(30);

    while(ros::ok()) {
        std_msgs::MultiArrayDimension dim;
        std_msgs::MultiArrayLayout layout;
        std_msgs::ByteMultiArray msg;

        dim.label = "something";
        dim.size = recv_num;
        dim.stride = recv_num;

        layout.dim.push_back(dim);
        layout.data_offset = 0;

        msg.layout = layout;

        struct sockaddr_in addr_client;
        size_t len = sizeof(addr_client);
        msg.data.resize(65535);

        recv_num = recvfrom(server_fd, msg.data.data(), 65535, 0, (struct sockaddr *)&addr_client, (socklen_t *)&len);

        if(recv_num > 0) {
            msg.data.resize(recv_num);
            publisher.publish(msg);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    close(server_fd);

    return 0;
}
 
