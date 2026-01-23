bygg om dessa 3 koder @generate_picking_list.py @generate_order_list.py @generate_graphical_picking_list.py 
använd nyaste data i detta fall product_sales_2024-02-12_to_2025-12-09 men räkna ut det senaste alltid från downloads enligt datumet i slutet på filnamnet.

@data/parametrar/Beställningsfrekvens.csv 
Till orderlist ska vi beakta beställningsfrekvens. Vi ska göra en lista för varje Leverantör med alla produkter som denna leverantör säljer och hur mycket som prognosticeras. Gör prediction lika långt som "Beställnings frekvens/antal dagar" så att för varje produkt från en leverantör med Beställnings frekvens/antal dagar 14 presenterar vi prognosen hur mycket vi kommer att få sålt totalt på 2 veckor i varje butik. Outputen från detta ska vara en dir orderlistor där vi har en csv fil för varje leverantör med alla produkter som kommer från denna leverantör med prognosticerad försäljning totalt för varje butik totalt för samma längd som beställningsfrekvens. 
Ha också saldo med som en kolumn samt stock_warning_limit som en kolumn. En kolumn ska heta beställningsbehov och den beräknas så att om vi får sålt enligt prognos så ska under tiden för beställningsfrekvens saldo inte gå under stock_warning_limit. stock_warning_limit multipliceras nu med antalet butiker som säljer produkten ifråga eftersom olika butiker har olika utbud. Produkternas närvaro per butik får man reda på från senaste stock_report csv
gör en csv fil parametrar.csv där användaren kan fylla i parametrar för hur länge det ska räcka det man levererar till en butik. Leveransfrekvens som en nummer i dagar per butik som går ut på att om nummern i dagar är n antal dagar så prognosticerar man totalförsäljningen per produkt n antal dagar framåt för den butiken och gör en plocklista som är uträknad enligt saldo och prognos att landa på stock_warning_limit efter 3 dagar om prognosen förverkligas.

deployment:
Läs igenom hela mitt kodspace och gör en exe fil för windows. När man kör exe filen ska först all data ledladdas 1download.py
sedan ska plocklistorna genereras 2generate_picking_list.py
sedan 3generate_order_list.py sedan 4generate_graphical_picking_list.py