const download = async function (form) {
    const res = await fetch("/api/download", {
        method: "POST",
        body: new FormData(form)
    })

    return res.json()
}

const process = async function (data) {
    const res = await fetch("/api/process", {
        method: "POST",
        body: JSON.stringify(data)
    })

    return res.json()
}

const downloadVideo = function (form) {
    spinner = document.getElementById("spinner")
    spinner.style.display = "unset"

    download(form).then(data => {
        console.log(data)

        process(data).then(res => {
            console.log(res)
            spinner.style.display = "none"
            window.location.href = `/upload?id=${res["id"]}&knn=${res["knn"]
                }&svm=${res["svm"]}&nn=${res["nn"]}`
        })
    })
}
