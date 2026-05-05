'use strict';

// sls invoke local --function getPlaylists --data '{"channelId": "UC-lHJZR3Gqxm24_Vd_AJ5Yw"}'

const fetch = require('node-fetch');
const AWS = require('aws-sdk');
const _ = require('lodash');
const config = require('./globals');

AWS.config.update({region: 'us-east-1'});

const ddb = new AWS.DynamoDB({apiVersion: '2012-08-10'});

module.exports.getPlaylists = async (event) => {
    if(Array.isArray(event)) {
        for(let j = 0; j < event.length; j++) {
            const channelId = event[j].channelId;
            const url = `${config.YT_PLAYLISTS_URL}${channelId}&maxResults=${config.YT_MAX_RESULTS}&key=${config.YT_TOKEN}`;
            const getData = async url => {
                try {
                    const response = await fetch(url);
                    const json = await response.json();
                    let dataNeeded = [];
        
                    _.forEach( json.items, (piece, key) => {
                        const { publishedAt, channelId, title, thumbnails, channelTitle } = piece.snippet;
                        
                        // Marshall
                        
                        const marshalled = AWS.DynamoDB.Converter.marshall({
                            channelId, publishedAt, title, thumbnails, playListId: piece.id, channelTitle, lastUpdatedAt: Date.now()
                        });
        
                        const params = {
                            TableName: config.AWS_DYNAMO_DB_TABLE,
                            Item: marshalled
                        };
        
                        dataNeeded.push(params);
                    });
                    return dataNeeded;
        
                } catch (error) {
                    console.log(error);
                }
            };
            let dataReady = await getData(url);
        
            for(let i = 0; i <= (dataReady.length - 1); i++) {
                let res = await ddb.putItem(dataReady[i]).promise();
                console.log('Res: ', i);
            }
        }
    }
}

module.exports.getVideosFromPlayList = async(event) => {
    if(event) {
        const playListId = event.pathParameters.playListId;
        
        let url = config.YT_PLAYLIST_VIDEOS + playListId;
        try {
            let resp = await fetch(url);
            let json = await resp.json();
            console.log('API response: ', json);
            return {
                body: JSON.stringify(json)
            }
        }
        catch(err) {
            console.log('An error occured while retrieving results: ', err)
            return {
                err
            }
        }
    }
}