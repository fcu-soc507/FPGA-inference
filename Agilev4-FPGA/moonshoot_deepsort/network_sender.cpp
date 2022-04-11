#include "network_sender.h"

#include <unistd.h>
#include <sys/types.h>   
#include <sys/socket.h>   
#include <netinet/in.h>   
#include <arpa/inet.h> 
#include <cstddef>
#include <chrono>
#include <string>
#include <sstream>

template<typename T> std::string to_bytes(T v) {
    std::string buf;
    auto pv = reinterpret_cast<char*>(&v);
    buf.resize(sizeof(v));
    std::copy(pv, pv+sizeof(T), buf.data());
    return buf;
}

NetworkSender::NetworkSender(const char* server_addr, int port)
    : server_addr_(server_addr)
    , port_(port)
    , sockfd(-1)
{
    connect();
}

NetworkSender::~NetworkSender() {
    if(sockfd > 0) { close(sockfd); }
}

int NetworkSender::connect() {
    struct sockaddr_in addr_serv;  
    int len;  
    memset(&addr_serv, 0, sizeof(addr_serv));  
    addr_serv.sin_family = AF_INET;  
    addr_serv.sin_addr.s_addr = inet_addr(server_addr_);  
    addr_serv.sin_port = htons(port_);
    len = sizeof(addr_serv);
    auto fd = socket(AF_INET, SOCK_DGRAM, 0);
    if(fd > 0) {
        sockfd = fd;
        return ::connect(fd, reinterpret_cast<sockaddr*>(&addr_serv), len);
    }
    return fd;
}

int NetworkSender::send_trackinfo(const ::TrackInfo& info) {
    std::stringstream ss;
    using namespace std::chrono;
    auto tp = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch());
    ss  << to_bytes(tp.count()) // 8byte
        << to_bytes(info.is_iframe) // 1byte
        << to_bytes(info.bboxes.size()); // 8byte

    for(const auto& box: info.bboxes) {
        ss  << to_bytes(box.label)
            << to_bytes(box.bbox.x)
            << to_bytes(box.bbox.y)
            << to_bytes(box.bbox.width)
            << to_bytes(box.bbox.height)
            << to_bytes(box.box_id);
    }
    if(sockfd > 0) {
        auto buf = ss.str();
        return send(sockfd, buf.data(), buf.size(), 0);
    }
    return -1;
}
