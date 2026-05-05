const fetch = require('node-fetch');
const AWS = require('aws-sdk');
const config = require('../globals');

module.exports.uname2ChannelId = async (event) => {
    
    if(event) {
        const channelUserName = event.pathParameters;
        const url = config.YT_CHANNEL_ID_URL + channelUserName.channel;
        const getData = async url => {
            try {
                let response = await fetch(url);
                let json = await response.json();
                if(json.items.length > 0) {
                    let channelId = json.items[0].id;
                    const channelUrl = config.YT_CHANNEL_INFO_URL + channelId;
                    try {
                        let channelResp = await fetch(channelUrl);
                        let channelJson = await channelResp.json();
                        return channelJson.items[0]
                    }
                    catch(err) {
                        return 'Channel info err: ' + err;
                    }
                }
                else {
                    return 'Channel ID does not exits';
                }
            }
            catch(err) {
                return 'Channel ID Error: ' + err;
            }
        }
        let resp = await getData(url);
        return {
            statusCode: 200,
            body: JSON.stringify(resp)
        }
    }
}