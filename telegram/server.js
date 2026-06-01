import { TelegramClient } from "telegram";
import {StoreSession, StringSession} from "telegram/sessions/index.js";
import readline from "readline";

// fill this  with the real values
const apiId =  0000000
const apiHash = ''
const pnumber= ''
const tfapass=''
// fill this  with the real values


//const stringSession = new StringSession(""); // fill this later with the value from session.save()
const storeSession = new StoreSession("./ses_tele");
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});
global.tgc=null;
(async () => {
    console.log("Loading interactive example...");
    global.tgc = new TelegramClient(storeSession, apiId, apiHash, {
        connectionRetries: 5,
    });
    await global.tgc.start({
        phoneNumber:pnumber,
        password: async ()=>{return new Promise((resolve)=>{resolve(tfapass)})},
        phoneCode: async () =>
            new Promise((resolve) =>
                rl.question("Please enter the code you received: ", resolve)
            ),
        onError: (err) => console.log(err),
    });
    console.log("You should now be connected.");
    console.log(global.tgc.session.save()); // Save this string to avoid logging in again
    await global.tgc.sendMessage("me", { message: "Hello!" });
})();
