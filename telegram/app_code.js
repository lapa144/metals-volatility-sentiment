import { TelegramClient } from 'telegram';
import { StoreSession, StringSession } from 'telegram/sessions/index.js';
import readline from 'readline';
import WebSocket, { WebSocketServer } from 'ws';
import express from 'express';
import http from 'http';
import { NewMessage } from "telegram/events/index.js";
import * as _ from 'underscore'
import { spawn } from 'child_process';
import { insertMessage, getUnprocessedMessages, markMessageAsProcessed, insertAnalysis } from './database.js';
/*should be replaced with real values*/
const apiId = 00000000;
const apiHash = '';
const pnumber = '';
const tfapass = '';
const storeSession = new StoreSession("./ses_tele");
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});
/*should be replaced with real values*/
global.tgc = null;
const monitoredChannels = [];
const wsClients = [];

(async () => {
    console.log("Loading interactive example...");
    // WebSocket setup
    const app = express();
    const server = http.createServer(app);
    const wss = new WebSocketServer({ server });

    app.use(express.static('./app/public')); // Serve HTML file

    wss.on('connection', (ws) => {
        wsClients.push(ws);
        ws.on('message', async (message) => {
            const { command, data } = JSON.parse(message);
            if (command === 'monitorChannels') {
                await monitorChannels(data.channels);
                ws.send(JSON.stringify({ command: 'status', data: 'Monitoring started' }));
            }
        });
    });

    server.listen(8888, () => {
        console.log('Server is listening on port 8080');
    });



    global.tgc = new TelegramClient(storeSession, apiId, apiHash, {
        connectionRetries: 5,
    });
    await global.tgc.start({
        phoneNumber: pnumber,
        password: async () => {
            return new Promise((resolve) => {
                resolve(tfapass);
            });
        },
        phoneCode: async () =>
            new Promise((resolve) =>
                rl.question("Please enter the code you received: ", resolve)
            ),
        onError: (err) => console.log(err),
    });
    console.log("You should now be connected.");
    console.log(global.tgc.session.save()); // Save this string to avoid logging in again


    // Monitor messages
    global.tgc.addEventHandler(handleNewMessage, new NewMessage({}));

})();

async function monitorChannels(channels) {
    channels.forEach(channel => {
        if (!monitoredChannels.includes(channel)) {
            monitoredChannels.push(channel);
        }
    });

    console.log(`Monitoring channels: ${monitoredChannels}`);
    // Fetch messages with delay
    await fetchMessagesWithDelay(monitoredChannels, 2000); // 2-second delay between requests
}

async function fetchMessagesWithDelay(channelIds, delay) {
    for (const channelId of channelIds) {
        try {
            console.log(`Fetching messages for channel ${channelId}`);
            const messages = await global.tgc.getMessages(channelId, { limit: 100 });
            console.log(messages)
            for (const message of messages) {
                console.log(`Inserting message with ID ${message.id}`);
                const messageId = await insertMessage(channelId, message);
                if (messageId) {
                    console.log(`Message inserted with ID ${messageId}`);
                    notifyClients(channelId, message.message);
                }
            }
        } catch (error) {
            console.error(`Failed to fetch messages for channel ${channelId}:`, error);
        }
        await new Promise(resolve => setTimeout(resolve, delay));
    }
}

async function handleNewMessage(event) {
    const message = event.message;
    const channelId = event.message.peerId.channelId;
    if (monitoredChannels.includes(channelId)) {
        console.log(`Handling new message with ID ${message.id}`);
        const messageId = await insertMessage(channelId, message);
        if (messageId) {
            console.log(`New message inserted with ID ${messageId}`);
            notifyClients(channelId, message.message);
        }
    }
}

function notifyClients(channelId, message) {
    wsClients.forEach(ws => {
        ws.send(JSON.stringify({ channelId, message }));
    });
}
