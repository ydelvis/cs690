'use strict';

module.exports.hello = async (event) => {
  if(Array.isArray(event)) {
    console.log('Is array', event[0].channelId);
  }
  else {
    console.log('Naada');
  }
};
