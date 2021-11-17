import { PythonShell } from 'python-shell';
PythonShell.run('script.py', null, function (err, data) {
  if (err) throw err;
  console.log(data);
});