#ifndef __NETWORK_SENDER_H__
#define __NETWORK_SENDER_H__

#include "track_info.h"

class NetworkSender {
public:
    NetworkSender(const char* server_addr, int port);
    ~NetworkSender();

    int connect();
    int send_trackinfo(const ::TrackInfo& info);

private:
    const char* server_addr_;
    int port_;
    int sockfd;
};


#endif