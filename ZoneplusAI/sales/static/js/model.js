
    const imageUrl = "{% static 'image/img1.png' %}";

    function runDev() {
        const dev = document.getElementById('showcase');
        const year = document.getElementById('year').value;
        const reg = document.getElementById('region').value;
        const prod = document.getElementById("category").value;
        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;

        if (year === "2016" && reg === "United States" && prod === "accessories" && age === "y" && gender === "m") {
            dev.innerHTML = `<img src="${imageUrl}" alt="2016" class="img-fluid">`;
            return false;
        }
    }

