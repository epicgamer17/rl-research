db = connect('mongodb://localhost:27017/admin');

db.createUser({
  user: 'ezra',
  pwd: mongoPassword,
  roles: [
    { role: 'userAdminAnyDatabase', db: 'admin' },
    { role: 'readWriteAnyDatabase', db: 'admin' },
  ],
});
