import sqlite3 from 'sqlite3';
const db = new sqlite3.Database('messages.db');

db.serialize(() => {
    console.log('Initializing database...');
    db.run(`CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id TEXT,
        message_id TEXT UNIQUE,
        message TEXT,
        message_object TEXT,
        processed INTEGER DEFAULT 0
    )`, (err) => {
        if (err) {
            console.error('Error creating messages table:', err);
        } else {
            console.log('Messages table created or already exists.');
        }
    });

    db.run(`CREATE TABLE IF NOT EXISTS analysis (
        message_id INTEGER,
        lang TEXT,
        topic TEXT,
        sentiment TEXT,
        assets TEXT,
        FOREIGN KEY(message_id) REFERENCES messages(id)
    )`, (err) => {
        if (err) {
            console.error('Error creating analysis table:', err);
        } else {
            console.log('Analysis table created or already exists.');
        }
    });

    db.run(`CREATE TABLE IF NOT EXISTS message_data (
        message_id INTEGER,
        date TEXT,
        sender_id TEXT,
        FOREIGN KEY(message_id) REFERENCES messages(id)
    )`, (err) => {
        if (err) {
            console.error('Error creating message_data table:', err);
        } else {
            console.log('Message data table created or already exists.');
        }
    });
});

export const insertMessage = async (channelId, message) => {
    return new Promise((resolve, reject) => {
        db.run(`INSERT INTO messages (channel_id, message_id, message, message_object) VALUES (?, ?, ?, ?)`,
            [channelId, message.id.toString(), message.message, JSON.stringify(message)], function (err) {
                if (err) {
                    if (err.code === 'SQLITE_CONSTRAINT') {
                        // Message already exists
                        console.log('Message already exists:', message.id);
                        return resolve(null);
                    }
                    console.error('Error inserting message:', err);
                    return reject(err);
                }
                const messageId = this.lastID;
                db.run(`INSERT INTO message_data (message_id, date, sender_id) VALUES (?, ?, ?)`,
                    [messageId, message.date, message.fromId==null?"-1":message.fromId.toString()], (err) => {
                        if (err) {
                            console.error('Error inserting message data:', err);
                            return reject(err);
                        }
                        resolve(messageId);
                    });
            });
    });
};

export const getUnprocessedMessages = () => {
    return new Promise((resolve, reject) => {
        db.all(`SELECT * FROM messages WHERE processed = 0`, (err, rows) => {
            if (err) {
                console.error('Error fetching unprocessed messages:', err);
                return reject(err);
            }
            resolve(rows);
        });
    });
};

export const markMessageAsProcessed = (id) => {
    return new Promise((resolve, reject) => {
        db.run(`UPDATE messages SET processed = 1 WHERE id = ?`, [id], (err) => {
            if (err) {
                console.error('Error marking message as processed:', err);
                return reject(err);
            }
            resolve();
        });
    });
};

export const insertAnalysis = (messageId, lang, topic, sentiment, assets) => {
    return new Promise((resolve, reject) => {
        db.run(`INSERT INTO analysis (message_id, lang, topic, sentiment, assets) VALUES (?, ?, ?, ?, ?)`,
            [messageId, lang, topic, sentiment, assets], (err) => {
                if (err) {
                    console.error('Error inserting analysis:', err);
                    return reject(err);
                }
                resolve();
            });
    });
};
