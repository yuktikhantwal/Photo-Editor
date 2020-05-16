const express = require('express')
const multer = require('multer')
const path = require('path')
const fs = require('fs')
const spawn = require('child_process').spawn

const storage = multer.diskStorage({
    destination: './images/',
    filename: function(req,file,cb){
        cb(null,'content' + path.extname(file.originalname))
    }
})

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 1000000
    },
    fileFilter: function(req,file,cb){
        checkFileType(file,cb)
    }
}).single('image')

function checkFileType(file,cb){
    const filetypes = /jpeg|jpg/
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase()) 
    const mimetype = filetypes.test(file.mimetype)

    if(extname && mimetype){
        return cb(null,true)
    }
    else{
        cb('Error: Images only!')
    }
}

const app = express()

app.set('view engine','hbs')
app.set('views',__dirname+'/views')

app.use(express.static('./images'))

app.get('/',(req,res)=>{
    res.render('index')
})

app.post('/upload',(req,res)=>{
    upload(req,res,(err)=>{
        if(err){
            res.render('index',{
                msg: err
            })
        }
        else{
            if(req.file == undefined){
                res.render('index',{
                    msg: 'Error: No file selected!'
                })
            }
            else{
                res.render('styles',{
                    msg: 'File uploaded!',
                    file: `uploads/${req.file.filename}`
                })
            }
        }
    })
})

app.get('/style',(req,res)=>{
    const style_num = req.query.num

    content = './images/content.jpg'
    style = './images/style-' + style_num + '.jpg'

    fs.copyFile(content,'./scripts/content.jpg',(err)=>{
        if(err) throw err
    })
    fs.copyFile(style,'./scripts/style.jpg',(err)=>{
        if(err) throw err
    })

    const process = spawn('python',['./scripts/model.py'])
    process.stdout.on('data',(data)=>{
        console.log(data.toString())
    })

    res.render('result')
})

const port = 3007
app.listen(port, () => console.log(`Server started on port ${port}`))