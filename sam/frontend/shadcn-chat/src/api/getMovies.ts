const getData = async (input: string) => {
    const options = {
      method: "GET",
      headers: {
        accept: "application/json",
        Authorization: "Bearer ....",
      },
    };
  
    const response = await fetch(`http://127.0.0.1:8000/items/${input}`, options)
      .then((response) => response.json())
      .catch((err) => {
        console.error(err);
        return null;
      });
  
    return response;
  };
  
  export default async function getMovies(input: string) {
    const data = await getData(input);
    return data;
  }