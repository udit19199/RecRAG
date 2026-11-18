Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=7f3cf4d9630016ac ===
Gold sentences and facts:
Sentence 0: In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers . When Maximoff throws the explosion into the sky with telekinesis , it damages a nearby building , killing several Wakandan humanitarian workers .
  Fact 0.0: In 1991, the brainwashed super-soldier James 'Bucky' Barnes was dispatched from a Hydra base in Siberia.
  Fact 0.1: In 1991, James 'Bucky' Barnes was dispatched to intercept an automobile carrying a case of super-soldier serum.
  Fact 0.2: In the present day, Ultron's defeat in the nation of Sokovia at the hands of the Avengers happened approximately one year earlier.
  Fact 0.3: Ultron was defeated in the nation of Sokovia by the Avengers.
  Fact 0.4: In the present day, Steve Rogers, Natasha Romanoff, Sam Wilson, and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos.
  Fact 0.5: Brock Rumlow blows himself up.
  Fact 0.6: Brock Rumlow hoped to kill Steve Rogers by blowing himself up.
  Fact 0.7: Wanda Maximoff throws the explosion into the sky with telekinesis.
  Fact 0.8: The explosion damages a nearby building.
  Fact 0.9: The explosion killed several Wakandan humanitarian workers.

Retrieved items:
<item 1>
Captain America: Civil War
It is the sequel to 2011 's Captain America : The First Avenger and 2014 's Captain America : The Winter Soldier , and the thirteenth film in the Marvel
Cinematic Universe ( MCU ) . The film is directed by Anthony and Joe Russo , with a screenplay by Christopher Markus & Stephen McFeely , and features an
ensemble cast , including Chris Evans , Robert Downey Jr. , Scarlett Johansson , Sebastian Stan , Anthony Mackie , Don Cheadle , Jeremy Renner , Chadwick
Boseman , Paul Bettany , Elizabeth Olsen , Paul Rudd , Emily VanCamp , Tom Holland , Frank Grillo , William Hurt , and Daniel Brühl . In Captain America : Civil
War , disagreement over international oversight of the Avengers fractures them into opposing factions -- one led by Steve Rogers and the other by Tony Stark .
Development of Civil War began in late 2013 when Markus and McFeely began writing the screenplay , which borrows concepts from the 2006 comic book storyline ``
Civil War '' , while also focusing on story and character elements from the previous Captain America films to conclude the trilogy .
</item 1>

<item 2>
Captain America: Civil War
Captain America : Civil War - Wikipedia Captain America : Civil War Jump to : navigation , search Captain America : Civil War Theatrical release poster Directed
by Anthony Russo Joe Russo Produced by Kevin Feige Screenplay by Christopher Markus Stephen McFeely Based on Captain America by Joe Simon Jack Kirby Starring
Chris Evans Robert Downey Jr . Scarlett Johansson Sebastian Stan Anthony Mackie Don Cheadle Jeremy Renner Chadwick Boseman Paul Bettany Elizabeth Olsen Paul
Rudd Emily VanCamp Tom Holland Frank Grillo William Hurt Daniel Brühl Music by Henry Jackman Cinematography Trent Opaloch Edited by Jeffrey Ford Matthew Schmidt
Production company Marvel Studios Distributed by Walt Disney Studios Motion Pictures Release date April 12 , 2016 ( 2016 - 04 - 12 ) ( Dolby Theatre ) May 6 ,
2016 ( 2016 - 05 - 06 ) ( United States ) Running time 147 minutes Country United States Language English Budget $250 million Box office $1.153 billion Captain
America : Civil War is a 2016 American superhero film based on the Marvel Comics character Captain America , produced by Marvel Studios and distributed by Walt
Disney Studios Motion Pictures .
</item 2>

<item 3>
Captain America: Civil War
The next month , it was revealed that the film would be converted to 3D in post-production , and that Brühl would be playing Helmut Zemo . Additionally ,
Elizabeth Olsen revealed she would reprise her role in the film as Wanda Maximoff / Scarlet Witch . Filming ( edit ) Film set for Captain America : Civil War in
Downtown Atlanta Principal photography began on April 27 , 2015 , in Fayette County , Georgia at Pinewood Atlanta Studios , under the working title Sputnik .
Other filming locations in the Atlanta metropolitan area included the Buckhead district of Atlanta ; the Peachtree Christian Church in Midtown Atlanta ;
Downtown Atlanta , including an area near Philips Arena known as The Gulch , which served as the market in Lagos ; Norcross , Georgia ; Porsche 's headquarters
in Atlanta at Aerotropolis Atlanta , which served as the Avengers ' headquarters in the film ; and the Atlanta Civic Center , which served as the IFID (
Institute for Infectious Diseases ) laboratory in Lagos ; the latter 's interiors doubled as MIT , and a third part was used for a Berlin location .
</item 3>

<item 4>
Captain America: Civil War
Contents ( hide ) 1 Plot 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 3.4. 1 Visual effects 4 Music 5 Release 5.1
Marketing 5.1. 1 Merchandise 5.2 Home media 6 Reception 6.1 Box office 6.1. 1 United States and Canada 6.1. 2 Other territories 6.2 Critical response 6.3
Accolades 7 Future 8 Notes 9 References 10 External links Plot ( edit ) In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a
Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in
the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a
biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers .
</item 4>

<item 5>
Graph query row:
work: Captain America: Civil War
relationship: RELATED_TO
place: Vienna
kind: city
</item 5>

<item 6>
Captain America: Civil War
Following positive reactions to test screenings of The Winter Soldier , the Russo brothers were brought back to direct in early 2014 . The film 's title and
premise were revealed in October 2014 , along with Downey 's casting ; additional cast members joined in the following months . Principal photography began in
April 2015 at Pinewood Atlanta Studios in Fayette County , Georgia , and continued in the Metro Atlanta area before concluding in Germany in August 2015 , with
the film being the first to use IMAX 's digital 2D cameras ( for the film 's central airport fight sequence ) . Visual effects were provided by nearly 20
different studios during the post-production process . Captain America : Civil War held its world premiere in Los Angeles on April 12 , 2016 , and was released
in the United States on May 6 , 2016 , in 3D and IMAX 3D . The film became a critical and commercial success , grossing over $1.1 billion worldwide , making it
the highest - grossing film of 2016 and , at the time , the 12th - highest - grossing film ever .
</item 6>

<item 7>
Captain America: Civil War
wikipedia.org/w/index.php?title=Captain_America:_Civil_War&oldid=819487324 '' Categories : 2016 films English - language films 2010s action thriller films 2010s
adventure films 2010s psychological thriller films 2010s science fiction films 2010s superhero films 2016 3D films American 3D films American action thriller
films American films American political thriller films American psychological thriller films American science fiction action films American science fiction
thriller films American sequel films Babelsberg Studio films Captain America films Crossover films Drone films Films scored by Henry Jackman Films about revenge
Films about terrorism Films based on works by Mark Millar Films directed by Anthony and Joe Russo Films set in 1991 Films set in Berlin Films set in Bucharest
Films set in Cleveland Films set in Germany Films set in Lagos Films set in London Films set in Massachusetts Films set in New York ( state ) Films set in
Queens , New York Films set in Siberia Films set in Vienna Films set in prison Films set in a fictional African country Films shot at Pinewood Atlanta Studios
Films shot in Atlanta Films shot in Berlin Films shot in Brazil Films shot in Germany Films shot in Norway Films shot
</item 7>

<item 8>
Captain America: Civil War
Filming locations in Germany included the Olympic Stadium in Berlin , and the Leipzig / Halle Airport in Schkeuditz . Additional filming also took place in
Puerto Rico and Norway , and was scheduled to in Iceland . Principal photography wrapped on August 22 , 2015 . Captain America : Civil War was the first film to
use IMAX 's digital 2D cameras , made in partnership with Arri , which was a customized version of the Arri Alexa 65 . According to Joe Russo , approximately
fifteen minutes of the film , the sequence where the two factions fight at Leipzig / Halle Airport , which was described as having `` incredible scale to it ''
and referred to by the directors as the `` splash panel '' , were shot with the cameras . Opaloch noted that there were discussions late in pre-production to
shoot the entire film with the Alexa 65 cameras . However , because of how close it was to the start of shooting as well as the idea being `` such a large -
scale undertaking '' , it was decided to only use them for the airport sequence , with the rest of film being shot on Arri Alexa XT digital cameras .
</item 8>

<item 9>
Captain America: Civil War
The film 's Southeast Asia premiere was held on April 21 at the Marina Bay Sands resort in Singapore , while the European premiere took place at Vue Cinemas in
Westfield London on April 26 . The film was released internationally starting from April 27 , releasing in 61 countries its first weekend , including the United
Kingdom on April 29 . The North America release on May 6 , took place in over 4,200 theaters , of which 3,300 were in 3D , along with 378 IMAX theaters , 480
premium large - format , and 161 D - Box locations . Internationally , the film opened in 955 IMAX theaters , while South Korea saw Civil War open in an ``
unprecedented '' 1,989 theaters . In September 2014 , TNT acquired the cable broadcast rights for Captain America : Civil War to air two years after its
theatrical release . Marketing ( edit ) In early July 2015 , Marvel began a viral marketing campaign for Ant - Man featuring Leslie Bibb , reprising her role
from the Iron Man films as journalist Christine Everhart , reporting for the faux news program , WHIH Newsfront .
</item 9>

<item 10>
Graph query row:
work: Captain America: Civil War
relationship: RELATED_TO
place: Lagos
kind: city
</item 10>
=== END ITEM id=7f3cf4d9630016ac ===

=== ITEM id=77e60381b640a418 ===
Gold sentences and facts:
Sentence 0: No . overall No. in series Title Vehicles tested Special guests Original air date 14 `` Past , Present , or Future '' Lamborghini Aventador S , Honda NSX , Rimac Concept One David Hasselhoff , Ricky Wilson 8 December 2017 ( 2017 - 12 - 08 ) The tent is now in the Cotswolds and will remain there for the duration of the series . To determine whether the past , present , or future of motoring is best , the presenters visit Switzerland with Clarkson in a petrol powered Lamborghini Aventador S representing the past , May in a hybrid powered Honda NSX representing the present , and Hammond in an all - electric Rimac Concept One representing the future . After starting at the top of the Gotthard Pass , Hammond brings the trio to a stay at the Park Weggis ( de ) health retreat and visit the Swiss Museum of Transport and other museums in Lucerne close to a charging station for the Rimac car . May and Clarkson reject this and go to a hill climbing race at Hemberg instead . As Clarkson can not fit into the Aventador whilst wearing a mandatory racing helmet , Clarkson sends a Lamborghini test driver in his place , thus setting the fastest time of all three . Hammond crashes after completing the course , destroying the Rimac in a fire . Footage taken from YouTube shows the wreckage being extinguished and Hammond being airlifted by the Swiss Air - Rescue ( REGA ) to hospital . The presenters discuss some inapropriate new features within cars on Conversation Street . Celebrity Brain Crash is replaced by Celebrity Face Off , in which Ricky Wilson beats David Hasselhoff to determine who the `` fastest former talent show host '' is . 15 `` The Falls Guys '' Ford GT , Mercedes AMG GT R Kevin Pietersen , Brian Wilson 15 December 2017 ( 2017 - 12 - 15 ) The presenters race from Central Park in Manhattan , New York City , to an observation tower overlooking the Niagara Falls . Clarkson drives the new Ford GT whilst May and Hammond ( on crutches ) use public transport , taking the New York City Subway , AirTrain JFK , a JetBlue Embraer E190 flight , plus two Niagara Frontier Transportation Authority buses . Despite Hammond and May 's flight arriving 15 minutes early , Clarkson narrowly wins the race . For Conversation Street the presenters discuss sponsorship ideas . Clarkson reviews the Mercedes AMG GT R , after which new test driver , Abbie Eaton , takes it for a timed lap around the Eboladrome . In `` Celebrity Face Off '' , Kevin Pietersen of South Africa beats Brian Wilson of the U.S. as the `` fastest person who makes a living throwing and hitting and catching small balls '' . 16 `` Bah Humbug - atti '' Kia Stinger GT , Bugatti Chiron Hugh Bonneville , Casey Anderson 22 December 2017 ( 2017 - 12 - 22 ) The presenters begin with Christmas tree lights made from vehicle parts and , Starting with Scalextric , they exchange humorous and inappropriate Christmas gifts in Conversation Street . May travels to Majorca to race a Kia Stinger GT against a pair of longboard riders along a mountain road , with May winning the race despite him racing uphill and the riders downhill . Hammond and May attempt to relieve lunchtime boredom by inventing `` lunchtime office car park racing '' at Houghton Hall business park in the Luton / Dunstable Urban Area . After brief interviews in which both share their own ownership of a Tesla Model S , Hugh Bonneville , representing the UK , and Casey Anderson , representing the United States , compete during `` Celebrity Face Off '' as the `` fastest person who lives with a bear '' , with Anderson winning by setting the faster lap time . Clarkson experiments with the jet set lifestyle , starting in the morning from the Da Vinci super-yacht moored in Saint - Tropez and then driving a Bugatti Chiron to L'Alpe De Venosc France for skiing at lunchtime , stopping briefly for a drag race against a BMW M4 at Gap -- Tallard Airport to give viewers a perspective on how fast the Chiron is . After leaving the Alps , Clarkson then pays an homage to the 1976 film C'était un rendez - vous , by speeding across Turin and reaching the Piazza Carignano ( it ) in 9 minutes . 17 `` Unscripted '' McLaren 720S , Audi TT RS , Ariel Nomad Michael Ball , Alfie Boe , Silvija Jurin 29 December 2017 ( 2017 - 12 - 29 ) The presenters begin by having a year - ending awards ceremony in Conversation Street . Hammond heads to the Eboladrome to test the McLaren 720S after which Eaton sets a lap time . Since The Grand Tour has been criticised for being too `` scripted '' , the presenters head to Zadar county and the island of Pag in Croatia to make a completely `` unscripted '' film . This results in Clarkson turning up in the Audi TT RS , Hammond turning up in the not at all comparable Ariel Nomad , and May turning up in an old Lada Riva which he turns into a fire engine . They go their separate ways though Clarkson and Hammond end up comparing their cars in a drag race , a rally stage ( both of which are won by Clarkson ) , visiting Maslenica , the Plitvice Lakes National Park , and following the D106 road until running out of road at Žigljen ferry port . Due to a lack of fires , May has to setup his own ones until Clarkson and Hammond setup one that proves too difficult for May , which the real fire brigade manage to put out with a firefighting aircraft , destroying May 's Lada fire engine in the process . Michael Ball and Alfie Boe compete in `` Celebrity Face Off '' with Ball setting the faster time and becoming the `` fastest classical singer with some kind of interesting connection to the British motor industry '' . 18 5 `` Up , Down and Round the Farm '' Ripsaw EV2 , Volkswagen Up ! GTI Bill Bailey , Dominic Cooper , Mark Higgins 5 January 2018 ( 2018 - 01 - 05 ) The presenters introduce some of the new cars for 2018 in Conversation Street . Hammond heads to Dubai to test a Sandrail dune buggy in the desert , followed by the Ripsaw `` high - end luxury tank '' , visiting the Dubai Mall . May reviews the Volkswagen Up ! GTI at the Eboladrome and in thick fog Eaton records a record slow time . Guests Bill Bailey and Dominic Cooper discuss their shared ownership of old French cars before racing each other in greasy / wet conditions with Cooper becoming the `` fastest person who has a failed former career in a band '' . Clarkson creates a Ken Block - style `` Farmkhana '' film using an old Subaru Impreza WRX STi around Hare Farm , near Rye , with bullet time photography . Hammond and May show the audience that Clarkson 's driving was mostly fake by sharing a `` making - of '' consisting of outtakes and behind - the - scenes footage featuring rally driver Mark Higgins , producer Phil Churchward , plus camera operator Ben Joiner breaking the fourth wall . Clarkson suggests that the whole thing is like sawing a woman in half during a magic show , and that really the audience do n't want to know . 19 6 `` Jaaaaaaaags '' None Luke Evans , Kiefer Sutherland 12 January 2018 ( 2018 - 01 - 12 ) The presenters head to the area around Grand Junction , Colorado , staying at the Gateway Canyons Resort & Spa , and eventually travelling to Telluride , Colorado with a variety of old Jaguar cars to establish the manufacturer 's reputation for reliability with Clarkson buying an XJR , May taking an XK8 convertible and Hammond bringing a Mark X . During the course of the trip the cars are driven around a dirt track ( where Hammond 's car is badly damaged , and is replaced by an XJ6 ) , along the runway at Telluride Regional Airport , ( where May 's car is destroyed , and is replaced by an XJS ) and down the slopes of the Telluride Ski Resort . At Hanging Flume overlook , Clarkson experiences a traffic stop by a Mesa County police officer and reads the `` Her Britannic Majesty '' text from his British passport . Despite Clarkson 's car being the best , May admitting his car had poor brakes deemed the challenge pointless . The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street . Kiefer Sutherland beats Luke Evans to become `` the fastest actor with a middle name of George '' . 20 7 `` It 's a Gas , Gas , Gas '' Lamborghini Huracán Performante Anthony Joshua , Bill Goldberg 19 January 2018 ( 2018 - 01 - 19 ) The presenters discuss some new supercars and how handbrake turns impress women within the Conversation Street Segment . Hammond tests the Lamborghini Huracán Performante at the Eboladrome beating May 's Ferrari 458 Speciale in a drag race before Eaton takes it round the track setting a fast time . May and Hammond then look at ingenious ways of filling up with fuel instead of stopping at a fuel station . Anthony Joshua becomes `` the fastest man who makes a living out of punching or strangling someone '' beating Bill Goldberg , with both previously owning cars from General Motors . Clarkson then talks about the Rivalry in rallying between Audi with the Quattro and Lancia with the 037 . They conclude with a `` terrible disappointment '' that Lancia is no longer around . 21 8 `` Blasts from the Past '' Jaguar XK SS , Aston Martin DB4 GT , Honda Civic Type R , Ford GT Stewart Copeland , Nick Mason 26 January 2018 ( 2018 - 01 - 26 ) The presenters head to the City of Pau , Pyrénées - Atlantiques , France , and home of the Pau Grand Prix on the Circuit de Pau - Ville ( fr ) street course . At Palais Beaumont ( fr ) Hammond arrives with a Jagaur XK SS and Clarkson in an Aston Martin DB4 GT ; both being modern builds of older designs from the 1950s `` back - catalogues '' of Jaguar and Aston Martin . They both race around the Streets of Pau before May turns up in a modern Honda Civic Type R to prove that new cars are better . All three depart to drive south over the Pyrenees mountain range , with Clarkson and Hammond experiencing difficulties in dense fog . Clarkson recants a previous incident from 2013 -- when Clarkson and Hammond 's driving licences were confiscated whilst filming The Perfect Road Trip in France -- and tries to goad May into speeding on the Autopista AP - 2 . At Barcelona the presenters race around the abandoned Autódromo de Sitges - Terramar and the extreme banking on its corners , with May winning , and Hammond abstaining following his history of crashes . During Conversation Street the presenters discuss names for gritters . At the Eboladrome , Clarkson retests the Ford GT following the Niagara Falls race , with Eaton setting a test lap . Nick Mason goes on to record a faster lap than Stewart Copeland to become `` the fastest drummer for a band beginning with the letter p '' . 22 9 `` Breaking , Badly '' Jaguar XJ220 , Bugatti EB 110 Super Sport Dynamo , Penn & Teller 2 February 2018 ( 2018 - 02 - 02 ) The presenters discuss new sport cars by various companies , jumpers and the speed of pigeons vs blackbirds during Conversation Street . Clarkson unsuccessfully sets out to prove that mankind has not significantly improved over the last 25 years by testing the Jaguar XJ220 and the Bugatti EB 110 Super Sport , the fastest supercars from the early 90s , both of which set slow times at the Eboladrome when driven by Eaton . The trio tries to break the UK speed record for amphibious vehicles by building one of their own . Clarkson oversees his idea of combining a Rolls - Royce Spey turbofan engine from a Nimrod jet and a speedboat with a Suzuki Jimny . However on a nearby lake , Clarkson and May have to abort their run after achieving only 3 mph ( 5 km / h ) and nearly sinking . Hammond and May then go on to build a vehicle of their own by combining a Bond Bug with the engine of a personal water craft nicknaming it the `` Pond Bug '' . Competing on Coniston Water during Coniston Power Boat Records Week , Clarkson , being the only one with a speedboat licence , operates the `` Pond Bug '' and achieves an average of 47.81 mph ( 76.94 km / h ) , setting a new UK record . After performing a series of magic tricks , Dynamo and Penn & Teller compete to become `` the fastest magician in the world '' , with Penn & Teller prevailing on a snowy track . 23 10 `` Oh , Canada '' Alfa Romeo Stelvio Quadrifoglio , Range Rover Velar P380 , Porsche Macan Turbo Performance Pack , Tesla Model X , Ford F150 Raptor Rory McIlroy , Paris Hilton 9 February 2018 ( 2018 - 02 - 09 ) The presenters head to Canada to prove that small SUV 's are useless with Clarkson picking an Alfa Romeo Stelvio Quadrifoglio , May choosing a Range Rover Velar P380 and Hammond taking a Porsche Macan Turbo Performance Pack . They are shown to be slower round the Area 27 circuit than a BMW M3 , not practical enough to carry Canadian Newfoundland dogs , and slower round a barrel course than a horse : concluding that SUVs are pointless . Despite this , executive producer Andy Wilman informs via text that they will race their SUVs whilst each towing a boat up a mountain . May abandons his Velar for his boat , which sinks , causing him to lose the race . Because Clarkson 's Stelvio has no tow bar he cheats and exchanges it for a Ford F150 Raptor , ultimately arriving first with the presenters concluding that the Raptor was better than all the SUVs being tested . On Conversation Street the presenters talk about new SUVs and vans . Clarkson reviews the Tesla Model X at the Eboladrome and on the road , accompanied by some lawyers , ostensibly to avoid Tesla pursuing any legal action , following the previous Tesla v. Top Gear court cases . Rory McIlroy and Paris Hilton discuss their car collections before McIlroy beats Hilton to become the `` World 's Fastest Golfing Enthusiast '' . 24 11 `` Feed the World '' None None 16 February 2018 ( 2018 - 02 - 16 ) The presenters head to Mozambique on a mission to feed the people of Bingo , an isolated town in the west of the country , with fish from the country 's capital of Maputo , covering a distance of approximately 200 miles . To do this , each of them attempt to find the best transport solution : May buys a Mercedes - Benz Estate and modifies it by fitting a Perspex tank filled with sea water in which to keep the fish alive ; Clarkson buys a Nissan Hardbody truck which he fits with a generator and an ice machine in an attempt to keep the fish frozen and Hammond buys a TVS Star motorcycle and devises a rack which he fits to the bike 's rear , inspired by a local method of drying and salting the fish to preserve them . After unsuccessfully trying to catch fish using local methods , they buy fish at a local market . Early in the journey , Clarkson 's truck crashes into May 's car upon rescuing it , damaging its radiator grille . Deeper into rural Mozambique , May constantly soaks himself with sea water from his ill - fitted tank and , due to his damaged radiator grille , suffers multiple breakdowns due to water intrusion . Hammond loses a large number of his fish on the bumpy roads , constantly gets stuck in thick mud and endlessly falls from his motorbike . Clarkson suffers very few issues except when his ice machine breaks , causing him to make small adjustments to his truck in order to smoke the fish with his exhaust . Upon May 's car breaking down in the middle of a lake , despite Hammond and Clarkson crossing successfully , Clarkson cuts off the front end of the Mercedes and tows what remains of the rear portion using his truck - much to May 's dismay and annoyance . Eventually , the three reach Bingo with May 's fish dead , Clarkson 's fish ruined and Hammond having almost no fish left . As a result , they sell no fish and leave Bingo in a helicopter `` like all charitable celebrities '' .
  Fact 0.0: Episode 14 is titled Past, Present, or Future.
  Fact 0.1: Episode 14 is numbered 14 overall and 14 in the series.
  Fact 0.2: Episode 14 tests the Lamborghini Aventador S, Honda NSX, and Rimac Concept One.
  Fact 0.3: Episode 14 has special guests David Hasselhoff and Ricky Wilson.
  Fact 0.4: Episode 14 originally aired on 8 December 2017.
  Fact 0.5: The tent is now in the Cotswolds and will remain there for the duration of the series.
  Fact 0.6: The presenters want to determine whether the past, present, or future of motoring is best.
  Fact 0.7: The presenters visit Switzerland with Clarkson in a petrol powered Lamborghini Aventador S representing the past.
  Fact 0.8: May drives a hybrid powered Honda NSX representing the present.
  Fact 0.9: Hammond drives an all-electric Rimac Concept One representing the future.
  Fact 0.10: After starting at the top of the Gotthard Pass, Hammond brings the trio to a stay at the Park Weggis (de) health retreat.
  Fact 0.11: Hammond visits the Swiss Museum of Transport and other museums in Lucerne.
  Fact 0.12: The Park Weggis stay is close to a charging station for the Rimac car.
  Fact 0.13: May and Clarkson reject the plan and go to a hill climbing race at Hemberg instead.
  Fact 0.14: Clarkson cannot fit into the Aventador whilst wearing a mandatory racing helmet.
  Fact 0.15: Clarkson sends a Lamborghini test driver in his place.
  Fact 0.16: The Lamborghini test driver sets the fastest time of all three.
  Fact 0.17: Hammond crashes after completing the course, destroying the Rimac in a fire.
  Fact 0.18: Footage taken from YouTube shows the wreckage being extinguished.
  Fact 0.19: Footage taken from YouTube shows Hammond being airlifted by the Swiss Air-Rescue (REGA) to hospital.
  Fact 0.20: The presenters discuss some inappropriate new features within cars on Conversation Street.
  Fact 0.21: Celebrity Brain Crash is replaced by Celebrity Face Off.
  Fact 0.22: In Celebrity Face Off, Ricky Wilson beats David Hasselhoff to determine who the fastest former talent show host is.
  Fact 0.23: Episode 15 is titled The Falls Guys.
  Fact 0.24: Episode 15 is numbered 15 overall and 15 in the series.
  Fact 0.25: Episode 15 tests the Ford GT and Mercedes AMG GT R.
  Fact 0.26: Episode 15 has special guests Kevin Pietersen and Brian Wilson.
  Fact 0.27: Episode 15 originally aired on 15 December 2017.
  Fact 0.28: The presenters race from Central Park in Manhattan, New York City, to an observation tower overlooking the Niagara Falls.
  Fact 0.29: Clarkson drives the new Ford GT.
  Fact 0.30: May and Hammond use public transport, with Hammond on crutches.
  Fact 0.31: May and Hammond take the New York City Subway, AirTrain JFK, a JetBlue Embraer E190 flight, and two Niagara Frontier Transportation Authority buses.
  Fact 0.32: Hammond and May's flight arrives 15 minutes early.
  Fact 0.33: Clarkson narrowly wins the race.
  Fact 0.34: For Conversation Street the presenters discuss sponsorship ideas.
  Fact 0.35: Clarkson reviews the Mercedes AMG GT R.
  Fact 0.36: After the review, new test driver Abbie Eaton takes the Mercedes AMG GT R for a timed lap around the Eboladrome.
  Fact 0.37: In Celebrity Face Off, Kevin Pietersen of South Africa beats Brian Wilson of the U.S. as the fastest person who makes a living throwing and hitting and catching small balls.
  Fact 0.38: Episode 16 is titled Bah Humbug - atti.
  Fact 0.39: Episode 16 is numbered 16 overall and 16 in the series.
  Fact 0.40: Episode 16 tests the Kia Stinger GT and Bugatti Chiron.
  Fact 0.41: Episode 16 has special guests Hugh Bonneville and Casey Anderson.
  Fact 0.42: Episode 16 originally aired on 22 December 2017.
  Fact 0.43: The presenters begin with Christmas tree lights made from vehicle parts.
  Fact 0.44: Starting with Scalextric, the presenters exchange humorous and inappropriate Christmas gifts in Conversation Street.
  Fact 0.45: May travels to Majorca to race a Kia Stinger GT against a pair of longboard riders along a mountain road.
  Fact 0.46: May wins the race despite him racing uphill and the riders downhill.
  Fact 0.47: Hammond and May attempt to relieve lunchtime boredom by inventing lunchtime office car park racing at Houghton Hall business park in the Luton / Dunstable Urban Area.
  Fact 0.48: After brief interviews, Hugh Bonneville and Casey Anderson each share their own ownership of a Tesla Model S.
  Fact 0.49: Hugh Bonneville represents the UK, and Casey Anderson represents the United States.
  Fact 0.50: Hugh Bonneville and Casey Anderson compete during Celebrity Face Off as the fastest person who lives with a bear.
  Fact 0.51: Casey Anderson wins by setting the faster lap time.
  Fact 0.52: Clarkson experiments with the jet set lifestyle.
  Fact 0.53: Clarkson starts the morning from the Da Vinci super-yacht moored in Saint-Tropez.
  Fact 0.54: Clarkson then drives a Bugatti Chiron to L'Alpe De Venosc France for skiing at lunchtime.
  Fact 0.55: Clarkson stops briefly for a drag race against a BMW M4 at Gap -- Tallard Airport to give viewers a perspective on how fast the Chiron is.
  Fact 0.56: After leaving the Alps, Clarkson pays an homage to the 1976 film C'etait un rendez-vous.
  Fact 0.57: Clarkson speeds across Turin and reaches the Piazza Carignano (it) in 9 minutes.
  Fact 0.58: Episode 17 is titled Unscripted.
  Fact 0.59: Episode 17 is numbered 17 overall and 17 in the series.
  Fact 0.60: Episode 17 tests the McLaren 720S, Audi TT RS, and Ariel Nomad.
  Fact 0.61: Episode 17 has special guests Michael Ball, Alfie Boe, and Silvija Jurin.
  Fact 0.62: Episode 17 originally aired on 29 December 2017.
  Fact 0.63: The presenters begin by having a year-ending awards ceremony in Conversation Street.
  Fact 0.64: Hammond heads to the Eboladrome to test the McLaren 720S.
  Fact 0.65: Eaton sets a lap time after Hammond tests the McLaren 720S.
  Fact 0.66: The Grand Tour has been criticised for being too scripted.
  Fact 0.67: The presenters head to Zadar county and the island of Pag in Croatia to make a completely unscripted film.
  Fact 0.68: Clarkson turns up in the Audi TT RS.
  Fact 0.69: Hammond turns up in the not at all comparable Ariel Nomad.
  Fact 0.70: May turns up in an old Lada Riva which he turns into a fire engine.
  Fact 0.71: The presenters go their separate ways.
  Fact 0.72: Clarkson and Hammond end up comparing their cars in a drag race and a rally stage, both of which are won by Clarkson.
  Fact 0.73: Clarkson and Hammond visit Maslenica, the Plitvice Lakes National Park, and follow the D106 road until running out of road at Zigljen ferry port.
  Fact 0.74: Due to a lack of fires, May has to set up his own ones.
  Fact 0.75: Clarkson and Hammond set up a fire that proves too difficult for May.
  Fact 0.76: The real fire brigade puts out the fire with a firefighting aircraft.
  Fact 0.77: The firefighting aircraft destroys May's Lada fire engine in the process.
  Fact 0.78: In Celebrity Face Off, Michael Ball sets the faster time against Alfie Boe.
  Fact 0.79: Michael Ball becomes the fastest classical singer with some kind of interesting connection to the British motor industry.
  Fact 0.80: Episode 18 is numbered 5 in the series and is titled Up, Down and Round the Farm.
  Fact 0.81: Episode 18 tests the Ripsaw EV2 and Volkswagen Up! GTI.
  Fact 0.82: Episode 18 has special guests Bill Bailey, Dominic Cooper, and Mark Higgins.
  Fact 0.83: Episode 18 originally aired on 5 January 2018.
  Fact 0.84: The presenters introduce some of the new cars for 2018 in Conversation Street.
  Fact 0.85: Hammond heads to Dubai to test a Sandrail dune buggy in the desert.
  Fact 0.86: Hammond then tests the Ripsaw high-end luxury tank.
  Fact 0.87: The presenters visit the Dubai Mall.
  Fact 0.88: May reviews the Volkswagen Up! GTI at the Eboladrome.
  Fact 0.89: In thick fog, Eaton records a record slow time.
  Fact 0.90: Guests Bill Bailey and Dominic Cooper discuss their shared ownership of old French cars.
  Fact 0.91: Bill Bailey and Dominic Cooper race each other in greasy and wet conditions.
  Fact 0.92: Dominic Cooper becomes the fastest person who has a failed former career in a band.
  Fact 0.93: Clarkson creates a Ken Block-style Farmkhana film using an old Subaru Impreza WRX STi around Hare Farm, near Rye.
  Fact 0.94: The Farmkhana film uses bullet time photography.
  Fact 0.95: Hammond and May show the audience that Clarkson's driving was mostly fake.
  Fact 0.96: They share a making-of consisting of outtakes and behind-the-scenes footage.
  Fact 0.97: The making-of features rally driver Mark Higgins, producer Phil Churchward, and camera operator Ben Joiner breaking the fourth wall.
  Fact 0.98: Clarkson suggests that the whole thing is like sawing a woman in half during a magic show.
  Fact 0.99: Clarkson suggests the audience do not really want to know.
  Fact 0.100: Episode 19 is numbered 6 in the series and is titled Jaaaaaaaags.
  Fact 0.101: Episode 19 has no vehicles tested.
  Fact 0.102: Episode 19 has special guests Luke Evans and Kiefer Sutherland.
  Fact 0.103: Episode 19 originally aired on 12 January 2018.
  Fact 0.104: The presenters head to the area around Grand Junction, Colorado, staying at the Gateway Canyons Resort & Spa.
  Fact 0.105: The presenters travel to Telluride, Colorado with a variety of old Jaguar cars.
  Fact 0.106: The trip establishes the manufacturer's reputation for reliability.
  Fact 0.107: Clarkson buys an XJR, May takes an XK8 convertible, and Hammond brings a Mark X.
  Fact 0.108: The cars are driven around a dirt track, where Hammond's car is badly damaged and is replaced by an XJ6.
  Fact 0.109: The cars are driven along the runway at Telluride Regional Airport, where May's car is destroyed and is replaced by an XJS.
  Fact 0.110: The cars are driven down the slopes of the Telluride Ski Resort.
  Fact 0.111: At Hanging Flume overlook, Clarkson experiences a traffic stop by a Mesa County police officer.
  Fact 0.112: Clarkson reads the Her Britannic Majesty text from his British passport.
  Fact 0.113: Despite Clarkson's car being the best, May admits his car had poor brakes.
  Fact 0.114: The presenters deem the challenge pointless.
  Fact 0.115: The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street.
  Fact 0.116: Kiefer Sutherland beats Luke Evans to become the fastest actor with a middle name of George.
  Fact 0.117: Episode 20 is numbered 7 in the series and is titled It's a Gas, Gas, Gas.
  Fact 0.118: Episode 20 tests the Lamborghini Huracan Performante.
  Fact 0.119: Episode 20 has special guests Anthony Joshua and Bill Goldberg.
  Fact 0.120: Episode 20 originally aired on 19 January 2018.
  Fact 0.121: The presenters discuss new supercars and how handbrake turns impress women within the Conversation Street segment.
  Fact 0.122: Hammond tests the Lamborghini Huracan Performante at the Eboladrome.
  Fact 0.123: Hammond beats May's Ferrari 458 Speciale in a drag race.
  Fact 0.124: Eaton takes the Lamborghini Huracan Performante round the track and sets a fast time.
  Fact 0.125: May and Hammond look at ingenious ways of filling up with fuel instead of stopping at a fuel station.
  Fact 0.126: Anthony Joshua becomes the fastest man who makes a living out of punching or strangling someone.
  Fact 0.127: Anthony Joshua beats Bill Goldberg.
  Fact 0.128: Anthony Joshua and Bill Goldberg both previously owned cars from General Motors.
  Fact 0.129: Clarkson talks about the rivalry in rallying between Audi with the Quattro and Lancia with the 037.
  Fact 0.130: The presenters conclude with a terrible disappointment that Lancia is no longer around.
  Fact 0.131: Episode 21 is numbered 8 in the series and is titled Blasts from the Past.
  Fact 0.132: Episode 21 tests the Jaguar XK SS, Aston Martin DB4 GT, Honda Civic Type R, and Ford GT.
  Fact 0.133: Episode 21 has special guests Stewart Copeland and Nick Mason.
  Fact 0.134: Episode 21 originally aired on 26 January 2018.
  Fact 0.135: The presenters head to the City of Pau, Pyrenees-Atlantiques, France, home of the Pau Grand Prix on the Circuit de Pau-Ville (fr) street course.
  Fact 0.136: At Palais Beaumont (fr), Hammond arrives with a Jaguar XK SS.
  Fact 0.137: Clarkson arrives at Palais Beaumont (fr) in an Aston Martin DB4 GT.
  Fact 0.138: Both cars are modern builds of older designs from the 1950s back-catalogues of Jaguar and Aston Martin.
  Fact 0.139: Hammond and Clarkson race around the Streets of Pau.
  Fact 0.140: May turns up in a modern Honda Civic Type R to prove that new cars are better.
  Fact 0.141: All three drive south over the Pyrenees mountain range.
  Fact 0.142: Clarkson and Hammond experience difficulties in dense fog.
  Fact 0.143: Clarkson recounts a 2013 incident in which Clarkson and Hammond's driving licences were confiscated whilst filming The Perfect Road Trip in France.
  Fact 0.144: Clarkson tries to goad May into speeding on the Autopista AP-2.
  Fact 0.145: At Barcelona, the presenters race around the abandoned Autodromo de Sitges-Terramar and its extreme banking on its corners.
  Fact 0.146: May wins the race at Autodromo de Sitges-Terramar.
  Fact 0.147: Hammond abstains from the race following his history of crashes.
  Fact 0.148: During Conversation Street the presenters discuss names for gritters.
  Fact 0.149: At the Eboladrome, Clarkson retests the Ford GT following the Niagara Falls race.
  Fact 0.150: Eaton sets a test lap.
  Fact 0.151: Nick Mason records a faster lap than Stewart Copeland.
  Fact 0.152: Nick Mason becomes the fastest drummer for a band beginning with the letter P.
  Fact 0.153: Episode 22 is numbered 9 in the series and is titled Breaking, Badly.
  Fact 0.154: Episode 22 tests the Jaguar XJ220 and Bugatti EB 110 Super Sport.
  Fact 0.155: Episode 22 has special guests Dynamo and Penn & Teller.
  Fact 0.156: Episode 22 originally aired on 2 February 2018.
  Fact 0.157: The presenters discuss new sport cars by various companies, jumpers, and the speed of pigeons vs blackbirds during Conversation Street.
  Fact 0.158: Clarkson unsuccessfully sets out to prove that mankind has not significantly improved over the last 25 years.
  Fact 0.159: The Jaguar XJ220 and Bugatti EB 110 Super Sport are the fastest supercars from the early 90s.
  Fact 0.160: Both the Jaguar XJ220 and Bugatti EB 110 Super Sport set slow times at the Eboladrome when driven by Eaton.
  Fact 0.161: The trio tries to break the UK speed record for amphibious vehicles by building one of their own.
  Fact 0.162: Clarkson oversees his idea of combining a Rolls-Royce Spey turbofan engine from a Nimrod jet and a speedboat with a Suzuki Jimny.
  Fact 0.163: On a nearby lake, Clarkson and May have to abort their run after achieving only 3 mph (5 km/h) and nearly sinking.
  Fact 0.164: Hammond and May build a vehicle by combining a Bond Bug with the engine of a personal water craft.
  Fact 0.165: They nickname the vehicle the Pond Bug.
  Fact 0.166: Competing on Coniston Water during Coniston Power Boat Records Week, Clarkson operates the Pond Bug.
  Fact 0.167: Clarkson is the only one with a speedboat licence.
  Fact 0.168: Clarkson achieves an average of 47.81 mph (76.94 km/h), setting a new UK record.
  Fact 0.169: After performing a series of magic tricks, Dynamo and Penn & Teller compete to become the fastest magician in the world.
  Fact 0.170: Penn & Teller prevail on a snowy track.
  Fact 0.171: Episode 23 is numbered 10 in the series and is titled Oh, Canada.
  Fact 0.172: Episode 23 tests the Alfa Romeo Stelvio Quadrifoglio, Range Rover Velar P380, Porsche Macan Turbo Performance Pack, Tesla Model X, and Ford F150 Raptor.
  Fact 0.173: Episode 23 has special guests Rory McIlroy and Paris Hilton.
  Fact 0.174: Episode 23 originally aired on 9 February 2018.
  Fact 0.175: The presenters head to Canada to prove that small SUVs are useless.
  Fact 0.176: Clarkson picks an Alfa Romeo Stelvio Quadrifoglio.
  Fact 0.177: May chooses a Range Rover Velar P380.
  Fact 0.178: Hammond takes a Porsche Macan Turbo Performance Pack.
  Fact 0.179: The SUVs are shown to be slower round the Area 27 circuit than a BMW M3.
  Fact 0.180: The SUVs are not practical enough to carry Canadian Newfoundland dogs.
  Fact 0.181: The SUVs are slower round a barrel course than a horse.
  Fact 0.182: The presenters conclude that SUVs are pointless.
  Fact 0.183: Executive producer Andy Wilman informs via text that they will race their SUVs whilst each towing a boat up a mountain.
  Fact 0.184: May abandons his Velar for his boat, which sinks, causing him to lose the race.
  Fact 0.185: Clarkson's Stelvio has no tow bar, so he cheats and exchanges it for a Ford F150 Raptor.
  Fact 0.186: Clarkson arrives first in the Ford F150 Raptor.
  Fact 0.187: The presenters conclude that the Raptor was better than all the SUVs being tested.
  Fact 0.188: On Conversation Street the presenters talk about new SUVs and vans.
  Fact 0.189: Clarkson reviews the Tesla Model X at the Eboladrome and on the road.
  Fact 0.190: Clarkson is accompanied by some lawyers to avoid Tesla pursuing any legal action.
  Fact 0.191: The lawyers follow the previous Tesla v. Top Gear court cases.
  Fact 0.192: Rory McIlroy and Paris Hilton discuss their car collections.
  Fact 0.193: McIlroy beats Hilton to become the World's Fastest Golfing Enthusiast.
  Fact 0.194: Episode 24 is numbered 11 in the series and is titled Feed the World.
  Fact 0.195: Episode 24 has no vehicles tested and no special guests.
  Fact 0.196: Episode 24 originally aired on 16 February 2018.
  Fact 0.197: The presenters head to Mozambique on a mission to feed the people of Bingo, an isolated town in the west of the country.
  Fact 0.198: The fish come from the country's capital of Maputo.
  Fact 0.199: The fish are carried a distance of approximately 200 miles.
  Fact 0.200: May buys a Mercedes-Benz Estate and fits a Perspex tank filled with sea water in which to keep the fish alive.
  Fact 0.201: Clarkson buys a Nissan Hardbody truck which he fits with a generator and an ice machine to keep the fish frozen.
  Fact 0.202: Hammond buys a TVS Star motorcycle and devises a rack which he fits to the bike's rear.
  Fact 0.203: The rack is inspired by a local method of drying and salting the fish to preserve them.
  Fact 0.204: After unsuccessfully trying to catch fish using local methods, they buy fish at a local market.
  Fact 0.205: Early in the journey, Clarkson's truck crashes into May's car upon rescuing it, damaging its radiator grille.
  Fact 0.206: Deeper into rural Mozambique, May constantly soaks himself with sea water from his ill-fitted tank.
  Fact 0.207: Due to his damaged radiator grille, May suffers multiple breakdowns due to water intrusion.
  Fact 0.208: Hammond loses a large number of his fish on the bumpy roads.
  Fact 0.209: Hammond constantly gets stuck in thick mud and endlessly falls from his motorbike.
  Fact 0.210: Clarkson suffers very few issues except when his ice machine breaks.
  Fact 0.211: Clarkson makes small adjustments to his truck in order to smoke the fish with his exhaust.
  Fact 0.212: When May's car breaks down in the middle of a lake, Hammond and Clarkson cross successfully.
  Fact 0.213: Clarkson cuts off the front end of the Mercedes and tows what remains of the rear portion using his truck, much to May's dismay and annoyance.
  Fact 0.214: The three reach Bingo with May's fish dead.
  Fact 0.215: Clarkson's fish are ruined.
  Fact 0.216: Hammond has almost no fish left.
  Fact 0.217: As a result, they sell no fish.
  Fact 0.218: They leave Bingo in a helicopter like all charitable celebrities.

Retrieved items:
<item 1>
The Grand Tour dropped the unpopular “The American” and “Celebrity Brain Crash” for series two, replacing them with Abbie Eaton as “Driver” and “Celebrity Face
Off,” where celebrities raced for lap times. Episodes featured Hammond’s Rimac crash and helicopter evacuation after a Swiss hill climb, Clarkson’s pneumonia,
and road-trip and car challenges. Celebrity Face Off guests included Ricky Wilson and David Hasselhoff, then Michael Ball and Alfie Boe. A 2017 episode
</item 1>

<item 2>
The cited articles cover *The Grand Tour*’s 2016–18 episodes, production, and reception. Season 2 dropped “Celebrity Brain Crash” and featured guests including
David Hasselhoff, Kiefer Sutherland, and Luke Evans. Reports noted Mustangs in the premiere and filming in Colorado, including a scene with Mesa County deputy
Conner Bell that was moved to Montrose County. Reviews described the debut as a bigger-budget, cinematic *Top Gear*; Amazon said the
</item 2>

<item 3>
The references cover reviews and reception of *The Grand Tour*, including divided reactions to its second episode, comparisons with *Top Gear*, and positive
reviews of season two’s premiere. They also document criticism of Richard Hammond’s 2016 “gay ice-cream” joke, a 2017 TRIC Award nomination, Rotten Tomatoes
coverage, and reports that season two’s first celebrity guests were all men.
</item 3>

<item 4>
The Grand Tour (TV series)
ASIN B0731ZQ3R7. 1 . Past , present or future Missing or empty url = ( help ) ; access - date = requires url = ( help ) ^ Jump up to : Starkey , Adam ( 13
December 2017 ) . `` The Grand Tour season 2 : Richard Hammond is on crutches in episode 2 after horrifying crash '' . Metro . London . Retrieved 13 December
2017 . second episode titled The Falls Guys , Clarkson drives ... Ford GT from New York City to Niagara Falls , with James May making the same journey via
public transport with Hammond ... Celebrity Face Off ... cricketer Kevin Pietersen and baseball 's Brian Wilson Jump up ^ `` The Grand Tour Season 2 Episode 3
'' . 16 December 2017 . Retrieved 16 December 2017 . ^ Jump up to : `` Zadarska manekenka Silvija oduševila u ' The Grand Touru ' '' ( Zadar model Silvija
delights in The Grand Tour ) . 24sata ( in Croatian ) . 3 January 2018 .
</item 4>

<item 5>
The Grand Tour (TV series)
The Daily Telegraph , while not as positive , still approved of the episode , stating `` The writing is still rather ropey . Clarkson 's suggestion of a new
nickname for May -- `` Dingleberry Handpump '' -- failed to raise a titter even among the super-fans gathered for the London premiere '' but also said that ``
for each wobble , there are just as many moments when The Grand Tour manages the clever trick Top Gear could pull off at its best : raising a chuckle while
sneaking in a bit of serious journalism at the same time . '' and ultimately gave the episode 3 out of 5 stars . Jeremy Clarkson himself believed that they had
`` hit the ground running with series 2 of the Grand Tour '' . Radio Times has criticised the gender balance of guests on the `` Celebrity Face Off '' segment .
There was only 1 woman guest in the episode `` Oh Canada '' ; Paris Hilton . Hothouse Flowers playing for the opening sequence of `` The Holy Trinity '' episode
in the Lucerne Valley , California Broadcast ( edit ) Australian free - to - air network Seven Network started broadcasting the first series of The Grand Tour
in mid-October 2017 .
</item 5>

<item 6>
The Grand Tour (TV series)
Rotten Tomatoes . Retrieved 17 July 2017 . Jump up ^ `` The Grand Tour season 2 episode 1 review : We 're in safe hands as Clarkson , Hammond and May return ''
. Digital Spy . Retrieved 8 December 2017 . Jump up ^ `` Television review : The Grand Tour on Amazon Prime '' . The Times . Retrieved 8 December 2017 . Jump up
^ `` The Grand Tour season 2 , episode 1 review : Has Jeremy Clarkson finally found his groove ? '' . The Daily Telegraph . Retrieved 8 December 2017 . Jump up
^ Clarkson , Jeremy ( 8 December 2017 ) . `` Jeremy Clarkson on Twitter '' . twitter.com . Retrieved 15 December 2017 . Jump up ^ Ling , Thomas ( 27 November
2017 ) . `` The Grand Tour 's first celebrity guests for series 2 have been revealed -- and there 's not a single woman '' . Retrieved 20 December 2017 . Jump
up ^ Long , Trevor .
</item 6>

<item 7>
The Grand Tour (TV series)
The Sunday Times : Driving . Retrieved 14 December 2017 . The show has a new test track in Enstone , ... there planning problems when ( Clarkson ) tried once
before to use the Enstone track , back in Top Gear days ... The noisier testing of fast cars will continue to be done in Wiltshire at RAF Wroughton ...
nicknamed `` the Eboladrome '' Jump up ^ Reed , Jason ( 24 January 2018 ) . `` The Secret Story Behind The Grand Tour 's Celebrity Braincrash '' . Shifting
Lanes . Retrieved 26 January 2018 . ^ Jump up to : McCreesh , Louise ( 27 November 2017 ) . `` The Grand Tour season 2 confirms celebrity guests including
Kiefer Sutherland and David Hasselhoff '' . Retrieved 19 December 2017 . The celebrity guests this series include Luke Evans , Kiefer Sutherland , Hugh
Bonneville , Kevin Pietersen and Dominic Cooper ( ... ) Dynamo , Rory Mcllroy , Michael Ball and Alfie Boe will also appear in season two .
</item 7>

<item 8>
*The Grand Tour* continued the *Top Gear* team’s motoring-entertainment style, following the 2015 departure of Jeremy Clarkson, Richard Hammond, James May and
producer Andy Wilman from the BBC show. Season two featured celebrity guests including Kiefer Sutherland, David Hasselhoff, Luke Evans, Rory McIlroy and Paris
Hilton; it also dropped its American “Stig” character. Testing took place at Enstone and RAF Wroughton, nicknamed the Ebol
</item 8>

<item 9>
The Grand Tour (TV series)
The Sun . Retrieved 23 December 2017 . ^ Jump up to : Daly , Emma ( 26 November 2016 ) . `` Jeremy Clarkson forced to ' kill off ' celebs on The Grand Tour
after BBC ' bans Top Gear rip - off star interviews ' '' . The Sun . Retrieved 26 November 2016 . Jump up ^ Lee , Kristen ( 16 November 2017 ) . `` ' Celebrity
Brain Crash ' Axed on the Grand Tour Season 2 '' . jalopnik.com . Retrieved 17 November 2017 . Jump up ^ McCreesh , Louise ( 27 November 2017 ) . `` The Grand
Tour season 2 confirms celebrity guests including The Hoff , Jack Bauer and Luke Evans '' . Digital Spy . Retrieved 28 November 2017 . Jump up ^ Hsieh , Tony (
20 November 2016 ) . `` The Grand Tour : 3 Mustangs from the Opening Sequence of the Premiere Episode '' . Grand Tour Nation . Retrieved 24 November 2016 . ^
Jump up to : `` Amazon Instant Video '' .
</item 9>

<item 10>
The Grand Tour (TV series)
Studio audience segments for the first series were filmed in the travelling tent between 17 July and 11 December 2016 in various countries , while for the
second series the tent was stationed permanently in the Cotswolds . Series one was filmed over the course of mid - to late 2016 and premiered on 18 November
2016 . Series two was filmed over the same time period the following year ; however , filming was set back due to various illnesses and injuries suffered by the
presenters . Due to this , it premiered on 8 December 2017 . In the days following the launch , `` The Holy Trinity '' became Amazon Video 's most watched
premiere episode . Overall , the show has received positive reviews from critics . Contents ( hide ) 1 Format 1.1 Films 1.2 Test tracks 1.2. 1 Eboladrome 1.2. 2
Enstone 1.3 Celebrity Face Off 1.4 Conversation Street 1.5 Running gags 1.6 Awards 2 Episodes 2.1 Series overview 2.2 Series 1 ( 2016 -- 17 ) 2.2.
</item 10>
=== END ITEM id=77e60381b640a418 ===

=== ITEM id=51972909d38fecf1 ===
Gold sentences and facts:
Sentence 0: Polar moment of inertia is a quantity used to predict an object 's ability to resist torsion , in objects ( or segments of objects ) with an invariant circular cross section and no significant warping or out - of - plane deformation . It is used to calculate the angular displacement of an object subjected to a torque . It is analogous to the area moment of inertia , which characterizes an object 's ability to resist bending and is required to calculate displacement .
  Fact 0.0: Polar moment of inertia is a quantity used to predict an object's ability to resist torsion.
  Fact 0.1: Polar moment of inertia predicts torsion resistance for objects, or segments of objects, with an invariant circular cross section.
  Fact 0.2: Polar moment of inertia predicts torsion resistance for objects with no significant warping or out-of-plane deformation.
  Fact 0.3: Polar moment of inertia is used to calculate the angular displacement of an object subjected to a torque.
  Fact 0.4: Polar moment of inertia is analogous to the area moment of inertia.
  Fact 0.5: The area moment of inertia characterizes an object's ability to resist bending.
  Fact 0.6: The area moment of inertia is required to calculate displacement.

Retrieved items:
<item 1>
Polar moment of inertia
See moment ( physics ) . Note : It has become common to use moment of inertia ( MOI ) to refer to either or both of the planar second moment of area , I = ∫ A x
2 d A ( \ displaystyle I = \ int _ ( A ) x ^ ( 2 ) \ , \ mathrm ( d ) A ) , where x is the distance to some reference plane , or the polar second moment of area
, I = ∫ A r 2 d A ( \ displaystyle I = \ int _ ( A ) r ^ ( 2 ) \ , \ mathrm ( d ) A ) , where r is the distance to some reference axis . In each case the
integral is over all the infinitesimal elements of area , dA , in some two - dimensional cross-section .
</item 1>

<item 2>
Polar moment of inertia
Englewood Cliffs , NJ . 1995 . ISBN 0 - 13 - 137589 - X . Jump up ^ http://www.efunda.com/math/areas/MomentOfInertia.cfm Jump up ^ Blodgeit . Design of Welded
Structures ( 1982 ) . Section 2.10. 2 . Jump up ^ `` What is the difference between the Polar Moment of Inertia , IPIP and the torsional constant , JTJT of a
cross section ? '' .
</item 2>

<item 3>
Polar moment of inertia
Conversion from area moment of inertia ( edit ) By the perpendicular axis theorem , the following equation relates I to the area moments of inertia about the
other two mutually perpendicular axes : I z = I x + I y ( \ displaystyle I_ ( z ) = I_ ( x ) + I_ ( y ) ) I x = I y ( \ displaystyle I_ ( x ) = I_ ( y ) ) is
only true for a circular or hollow circular section with a constant thickness . For non circular cross section analytical formulas are needed : torsion constant
. Second moment of inertia , I , is the resistance of a solid to being twisted . Polar moment of inertia , I or J , is the inertia moment of a rotating solid
Application ( edit ) The polar moment of area appears in the formulae that describe torsional stress and angular displacement .
</item 3>

<item 4>
Polar moment of inertia
I z = ∫ A ρ 2 d A ( \ displaystyle I_ ( z ) = \ int _ ( A ) \ rho ^ ( 2 ) \ , dA ) I = the polar moment of area about the axis z dA = an elemental area ρ = the
radial distance to the element dA from the axis z For a circular section with radius r : I z = ∫ 0 2 π ∫ 0 r ρ 2 ρ d ρ d φ = π r 4 2 ( \ displaystyle I_ ( z ) =
\ int _ ( 0 ) ^ ( 2 \ pi ) \ int _ ( 0 ) ^ ( r ) \ rho ^ ( 2 ) \ rho \ , d \ rho \ , d \ phi = ( \ frac ( \ pi r ^ ( 4 ) ) ( 2 ) ) ) Unit ( edit ) The SI unit
for polar moment of area , like the area moment of inertia , is meter to the fourth power ( m ) .
</item 4>

<item 5>
Polar moment of inertia
If one adds a factor of safety of 5 and re-calculates the radius with the maximal stress equal to the yield stress / 5 the result is a radius of 0.343 m , or a
diameter of 69 cm , the approximate size of a turboset shaft in a nuclear power plant . Comparing various moments of inertias for a cylinder ( edit ) Polar
moment of inertia ( edit ) Hollow cylinder I z = π ( D 4 − d 4 ) 32 ( \ displaystyle I_ ( z ) = ( \ frac ( \ pi \ left ( D ^ ( 4 ) - d ^ ( 4 ) \ right ) ) ( 32
) ) ) Mass moment of inertia J z = I z ρ l = π ρ l ( D 4 − d 4 ) 32 ( \ displaystyle J_ ( z ) = I_ ( z ) \ rho l = ( \ frac ( \ pi \ rho l \ left ( D ^ ( 4 ) -
d ^ ( 4 ) \ right ) ) ( 32 ) ) ) Solid cylinder I z = π D 4 32 ( \ displaystyle I_ ( z ) = ( \ frac
</item 5>

<item 6>
Polar moment of inertia
Moment of inertia is , strictly , the second moment of mass with respect to distance from an axis : I = ∫ m r 2 d m ( \ displaystyle I = \ int _ ( m ) r ^ ( 2 )
\ mathrm ( d ) m ) , where r is the distance to some potential rotation axis , and the integral is over all the infinitesimal elements of mass , dm , in a three
- dimensional space occupied by an object . The MOI , in this sense , is the analog of mass for rotational problems . Contents ( hide ) 1 Limitations 2
Definition 3 Unit 4 Conversion from Area Moment of Inertia 5 Application 6 Sample calculation 7 Comparing various moments of inertias for a cylinder 7.1 Polar
moment of inertia 8 See also 9 References 10 External links Limitations ( edit ) The polar moment of area can not be used to analyze shafts with non-circular
cross-sections . In such cases , the torsion constant can be substituted instead .
</item 6>

<item 7>
Polar moment of inertia
In such cases , the torsion constant can be substituted instead . In objects with significant cross-sectional variation ( along the axis of the applied torque )
, which can not be analyzed in segments , a more complex approach may have to be used . See 3 - D elasticity . However the polar moment of area can be used to
calculate the moment of inertia of an object with arbitrary cross-section . Definition ( edit ) A schematic showing how the polar moment of area is calculated
for an arbitrary shape about an axis o. ρ is the radial distance to the element dA .
</item 7>

<item 8>
Polar moment of inertia
Polar moment of inertia - wikipedia Polar moment of inertia Jump to : navigation , search Polar moment of inertia is a quantity used to predict an object 's
ability to resist torsion , in objects ( or segments of objects ) with an invariant circular cross section and no significant warping or out - of - plane
deformation . It is used to calculate the angular displacement of an object subjected to a torque . It is analogous to the area moment of inertia , which
characterizes an object 's ability to resist bending and is required to calculate displacement . The larger the polar moment of area , the less the beam will
twist , when subjected to a given torque . Polar moment of area should not be confused with moment of inertia , which characterizes an object 's angular
acceleration due to a torque . See moment ( physics ) .
</item 8>

<item 9>
Polar moment of inertia
. first1 = missing last1 = in Authors list ( help ) External links ( edit ) Torsion of Shafts - engineeringtoolbox.com Elastic Properties and Young Modulus for
some Materials - engineeringtoolbox.com Retrieved from `` https://en.wikipedia.org/w/index.php?title=Polar_moment_of_inertia&oldid=805920942 '' Categories :
Continuum mechanics Mechanics Hidden categories : CS1 errors : missing author or editor Talk Contents About Wikipedia Eesti 한국어 עברית Magyar Русский Simple
English Українська 中文 Edit links This page was last edited on 18 October 2017 , at 13 : 57 . About Wikipedia
</item 9>

<item 10>
Polar moment of inertia
= π D 4 32 ( \ displaystyle I_ ( z ) = ( \ frac ( \ pi D ^ ( 4 ) ) ( 32 ) ) ) Mass moment of inertia J z = I z ρ l = π ρ l D 4 32 ( \ displaystyle J_ ( z ) = I_
( z ) \ rho l = ( \ frac ( \ pi \ rho lD ^ ( 4 ) ) ( 32 ) ) ) where : d ( \ displaystyle d ) is the inner diameter in m D ( \ displaystyle D ) is the outside
diameter in m J z ( \ displaystyle J_ ( z ) ) is the mass moment of inertia in kg m l ( \ displaystyle l ) is the length of cylinder in m ρ ( \ displaystyle \
rho ) is the specific mass in kg / m See also ( edit ) Torsion constant Torsion spring Torsion bar Area moment of inertia References ( edit ) Jump up ^ Ugural
AC , Fenster SK . Advanced Strength and Applied Elasticity . 3rd Ed . Prentice - Hall Inc . Englewood Cliffs , NJ . 1995 .
</item 10>
=== END ITEM id=51972909d38fecf1 ===

=== ITEM id=583529420d5dc0ee ===
Gold sentences and facts:
Sentence 0: At the Council , following advice offered by Simon Peter ( Acts 15 : 7 -- 11 and Acts 15 : 14 ) , Barnabas and Paul gave an account of their ministry among the gentiles ( Acts 15 : 12 ) , and the apostle James quoted from the words of the prophet Amos ( Acts 15 : 16 -- 17 , quoting Amos 9 : 11 -- 12 ) . James added his own words to the quotation : `` Known to God from eternity are all His works '' and then submitted a proposal , which was accepted by the Church and became known as the Apostolic Decree :
  Fact 0.0: Simon Peter offered advice at the Council, recorded in Acts 15:7-11 and Acts 15:14.
  Fact 0.1: Barnabas and Paul gave an account of their ministry among the gentiles at the Council, recorded in Acts 15:12.
  Fact 0.2: The account given by Barnabas and Paul followed advice offered by Simon Peter.
  Fact 0.3: The apostle James quoted from the words of the prophet Amos, recorded in Acts 15:16-17.
  Fact 0.4: The apostle James quoted Amos 9:11-12.
  Fact 0.5: James added his own words to the quotation: 'Known to God from eternity are all His works'.
  Fact 0.6: James submitted a proposal.
  Fact 0.7: The proposal was accepted by the Church.
  Fact 0.8: The proposal became known as the Apostolic Decree.

Retrieved items:
<item 1>
Council of Jerusalem
463 ) . Dibelius ' view ( Studies in the Acts of the Apostles ( Engtr 1956 ) , pp. 93 -- 101 ) is that Luke 's treatment is literary - theological and can make
no claim to historical worth. `` , Mounce , `` Apostolic Council '' , in Bromiley ( ed . ) `` The International Standard Bible Encyclopedia '' , volume 1 , p.
200 ( rev . ed. 2001 ) . Wm . B. Eerdmans . Jump up ^ `` There is an increasing trend among scholars toward considering the Jerusalem Council as historical
event . An overwhelming majority identifies the reference to the Jerusalem Council in Acts 15 with Paul 's account in Gal. 2.1 -- 10 , and this accord is not
just limited to the historicity of the gathering alone but extends also to the authenticity of the arguments deriving from the Jerusalem church itself . '' ,
Philip , `` The Origins of Pauline Pneumatology : the Eschatological Bestowal of the Spirit '' , Wissenschaftliche Untersuchungen zum Neuen Testament 2 , Reihe
, p. 205 ( 2005 ) .
</item 1>

<item 2>
Council of Jerusalem
The council decided that Gentile converts to Christianity were not obligated to keep most of the Law of Moses , including the rules concerning circumcision of
males . The Council did , however , retain the prohibitions on eating blood , meat containing blood , and meat of animals not properly slain , and on
fornication and idolatry , sometimes referred to as the Apostolic Decree or Jerusalem Quadrilateral . Accounts of the council are found in Acts of the Apostles
chapter 15 ( in two different forms , the Alexandrian and Western versions ) and also possibly in Paul 's letter to the Galatians chapter 2 . Some scholars
dispute that Galatians 2 is about the Council of Jerusalem ( notably because Galatians 2 describes a private meeting ) while other scholars dispute the
historical reliability of the Acts of the Apostles . Contents ( hide ) 1 Historical background 2 Issues and outcome 3 Historicity 4 Interpreting the Council 's
decision 4.1 Modern interpretation 4.1.
</item 2>

<item 3>
Council of Jerusalem
, Paget , `` Jewish Christianity '' , in Horbury , et al. , `` The Cambridge History of Judaism : The Early Roman Period '' , volume 3 , p. 744 ( 2008 ) .
Cambridge University Press . Jump up ^ `` Paul 's account of the Jerusalem Council in Galatians 2 and the account of it recorded in Acts have been considered by
some scholars as being in open contradiction . '' , Paget , `` Jewish Christianity '' , in Horbury , et al. , `` The Cambridge History of Judaism : The Early
Roman Period '' , volume 3 , p. 744 ( 2008 ) . Cambridge University Press . Jump up ^ `` There is a very strong case against the historicity of Luke 's account
of the Apostolic Council '' , Esler , `` Community and Gospel in Luke - Acts : The Social and Political Motivations of Lucan Theology '' , p. 97 ( 1989 ) .
Cambridge University Press . Jump up ^ `` The historicity of Luke 's account in Acts 15 has been questioned on a number of grounds . ''
</item 3>

<item 4>
Council of Jerusalem
70 ) . Interpreting the Council 's decision ( edit ) See also : Paul the Apostle and Judaism and Early Christianity Part of a series on Jewish Christianity
Figures Jesus John the Baptist Simon Peter Twelve Apostles James , brother of Jesus Simeon of Jerusalem Jude Paul Ancient groups Ebionites Elcesaites Nazarenes
Pejoratives Judaizers Legalists Pauline Christianity Recent groups Hebrew Christian movement Messianic Judaism Saint Thomas Christians Adversity Split of
Christianity and Judaism Paul and Judaism Marcionism Christian anti-semitism Constantine Writings Gospel of Matthew Epistle of James Clementine Didache Liturgy
of St James Book of Elchasai Jewish -- Christian gospels Gospel of the Ebionites Gospel of the Hebrews Gospel of the Nazarenes Issues Aramaic of Jesus Yeshua (
name ) Council of Jerusalem Expounding of the Law Sabbath Quartodecimanism Noahide laws James 's Apostolic Decree was that most Mosaic law , including the
requirement for circumcision of males , was not obligatory for Gentile converts , possibly in order to make it easier for them to join the movement .
</item 4>

<item 5>
Council of Jerusalem
This is the reason why this Council was called by Jesus ' apostle James , who , according to Paul 's account in Galatians 2 : 10 concluded that Paul would not
have to demand his men to become circumcised , and according to Luke 's account in Acts 15 : 13 -- 21 , James not only permitted uncircumcised men to remain in
the group , but he said in closing the Council ( Acts 15 : 19 ) `` We should not trouble the Gentiles who are turning to God . '' As for Orthodox Judaism today
, the rabbis do not believe that Gentiles need to be circumcised , unless they wish to convert to Judaism , which is discouraged . Instead , Gentiles need only
follow the Seven Laws of Noah to be assured of a place in the `` World to Come '' .
</item 5>

<item 6>
Council of Jerusalem
, Paget , `` Jewish Christianity '' , in Horbury , et al. , `` The Cambridge History of Judaism : The Early Roman Period '' , volume 3 , p. 744 ( 2008 ) .
Cambridge University Press . Jump up ^ `` However , numerous scholars have challenged the historicity of the Jerusalem Council as related by Acts , Paul 's
presence there in the manner that Luke describes , the issue of idol - food being thrust on Paul 's Gentile mission , and the historical reliability of Acts in
general . '' , Fotopolous , `` Food Offered to Idols in Roman Corinth : a socio - rhetorical reconsideration '' , pp. 181 -- 182 ( 2003 ) . Mohr Siebeck . Jump
up ^ `` Sahlin rejects the historicity of Acts completely ( Der Messias und das Gottesvolk ( 1945 ) ) . Haenchen 's view is that the Apostolic Council `` is an
imaginary construction answering to no historical reality '' ( The Acts of the Apostles ( Engtr 1971 ) , p. 463 ) .
</item 6>

<item 7>
Council of Jerusalem
At the Council , following advice offered by Simon Peter ( Acts 15 : 7 -- 11 and Acts 15 : 14 ) , Barnabas and Paul gave an account of their ministry among the
gentiles ( Acts 15 : 12 ) , and the apostle James quoted from the words of the prophet Amos ( Acts 15 : 16 -- 17 , quoting Amos 9 : 11 -- 12 ) . James added his
own words to the quotation : `` Known to God from eternity are all His works '' and then submitted a proposal , which was accepted by the Church and became
known as the Apostolic Decree : It is my judgment , therefore , that we should not make it difficult for the Gentiles who are turning to God . Instead we should
write to them , telling them to abstain from food polluted by idols , from sexual immorality , from the meat of strangled animals and from blood . For the law
of Moses has been preached in every city from the earliest times and is read in the synagogues on every Sabbath .
</item 7>

<item 8>
Council of Jerusalem
Authorship and development ( show ) Authorship Dating Hebrew canon Old Testament canon New Testament canon Documentary hypothesis Mosaic authorship Pauline
epistles Petrine epistles Johannine works Translations and manuscripts ( show ) Samaritan Torah Dead Sea scrolls Masoretic Text Targumim Peshitta Septuagint
Vulgate Gothic Bible Vetus Latina Luther Bible English Bibles Biblical studies ( show ) Archeology Artifacts Dating Historicity Internal consistency People
Places Names Novum Testamentum Graece Documentary hypothesis Synoptic problem NT textual categories Biblical criticism Historical Textual Source Form Redaction
Canonical Interpretation ( show ) Hermeneutics Pesher Midrash Pardes Allegorical interpretation Literalism Prophecy Inspiration Perspectives ( show ) Gnostic
Islamic Qur'anic Muhammad Inerrancy Infallibility Criticism of the Bible Outline of Bible - related topics Bible book Bible portal The Council of Jerusalem or
Apostolic Council was held in Jerusalem around AD 50 . It is unique among the ancient pre-ecumenical councils in that it is considered by Catholics and Orthodox
to be a prototype and forerunner of the later ecumenical councils and a key part of Christian ethics .
</item 8>

<item 9>
Council of Jerusalem
In the view of some scholars , the reminder of James and the elders here is an expression of concern that Paul was not fully teaching the decision of the
Jerusalem Council 's letter to Gentiles , particularly in regard to non-strangled kosher meat , which contrasts with Paul 's advice to Gentiles in Corinth , to
`` eat whatever is sold in the meat markets '' ( 1 Corinthians 10 : 25 ) . Historicity ( edit ) See also : Historical reliability of the Acts of the Apostles
The description of the Apostolic Council in Acts 15 , generally considered the same event described in Galatians 2 , is considered by some scholars to be
contradictory to the Galatians account . The historicity of Luke 's account has been challenged , and was rejected completely by some scholars in the mid to
late 20th century . However , more recent scholarship inclines towards treating the Jerusalem Council and its rulings as a historical event , though this is
sometimes expressed with caution .
</item 9>

<item 10>
Council of Jerusalem
Council of Jerusalem - wikipedia Council of Jerusalem Jump to : navigation , search This article is about the first century Council of early Christianity . For
other uses , see Jerusalem Council . Council of Jerusalem Date c. 50 Accepted by most Christian denominations Next council Ancient church councils (
pre-ecumenical ) President unspecified , but presumably James the Just and perhaps Simon Peter Topics Controversy of circumcision and the validity of the Law of
Moses . Documents and statements Excerpts from New Testament ( Acts of Apostles and perhaps Epistle to the Galatians ) . Chronological list of Ecumenical
councils Part of a series on Ecumenical councils of the Catholic Church Renaissance depiction of the Council of Trent Antiquity ( c.
</item 10>
=== END ITEM id=583529420d5dc0ee ===

=== ITEM id=53228d335868e46f ===
Gold sentences and facts:
Sentence 0: A main vertical stroke is called a stem . The letter m has three , the left , middle , and right stems . The central stroke of an s is called the spine . When the stroke is part of a lowercase and rises above the height of an x ( called the x height ) , it is called an ascender . Letters with ascenders are bdfhkl . A stroke which drops below the baseline is a descender . Letters with descenders are gjpqy . An arching stroke is called a shoulder as in the top of an R or sometimes just an arch , as in hnm . A closed curved stroke is called a bowl in bdopq DOPQ ; B has two bowls . A trailing outstroke , as in jy JQR is called a tail . The inferior diagonal stroke in K is called leg . A short horizontal stroke , as in the center of ef and the middle stroke of EF , is called a bar . When the strokes connect as in A and H or cross strokes as in t is also known as crossbar . A longer horizontal stroke at the top or bottom , as in ET , is called an arm . The bottom of the two - story g is called a loop ; the very short stroke at the top is called the ear . ij each have a dot , jot , or tittle . Angles of strokes are called apices if at the top and vertices if at the bottom . w has one apex and two vertices ; v has one vertex .
  Fact 0.0: A main vertical stroke is called a stem.
  Fact 0.1: The letter m has three stems: the left stem, the middle stem, and the right stem.
  Fact 0.2: The central stroke of an s is called the spine.
  Fact 0.3: When a stroke is part of a lowercase letter and rises above the height of an x (called the x height), it is called an ascender.
  Fact 0.4: The letters with ascenders are b, d, f, h, k, and l.
  Fact 0.5: A stroke which drops below the baseline is called a descender.
  Fact 0.6: The letters with descenders are g, j, p, q, and y.
  Fact 0.7: An arching stroke is called a shoulder, as in the top of an R.
  Fact 0.8: An arching stroke is sometimes just called an arch, as in h, n, and m.
  Fact 0.9: A closed curved stroke is called a bowl, as in b, d, o, p, q, D, O, P, and Q.
  Fact 0.10: The letter B has two bowls.
  Fact 0.11: A trailing outstroke, as in j, y, J, Q, and R, is called a tail.
  Fact 0.12: The inferior diagonal stroke in K is called a leg.
  Fact 0.13: A short horizontal stroke, as in the center of e and f and the middle stroke of E and F, is called a bar.
  Fact 0.14: When strokes connect, as in A and H, or cross, as in t, the stroke is also known as a crossbar.
  Fact 0.15: A longer horizontal stroke at the top or bottom, as in E and T, is called an arm.
  Fact 0.16: The bottom of the two-story g is called a loop.
  Fact 0.17: The very short stroke at the top of the two-story g is called the ear.
  Fact 0.18: The letters i and j each have a dot, jot, or tittle.
  Fact 0.19: Angles of strokes are called apices if at the top.
  Fact 0.20: Angles of strokes are called vertices if at the bottom.
  Fact 0.21: The letter w has one apex and two vertices.
  Fact 0.22: The letter v has one vertex.

Retrieved items:
<item 1>
Typeface anatomy
Typeface Anatomy - wikipedia Typeface Anatomy Jump to : navigation , search Typographic parts of a glyph : 1 ) x-height ; 2 ) ascender line ; 3 ) apex ; 4 )
baseline ; 5 ) ascender ; 6 ) crossbar ; 7 ) stem ; 8 ) serif ; 9 ) leg ; 10 ) bowl ; 11 ) counter ; 12 ) collar / link / neck ; 13 ) loop ; 14 ) ear ; 15 ) tie
; 16 ) horizontal bar ; 17 ) arm ; 18 ) vertical bar ; 19 ) cap height ; 20 ) descender line . Typeface anatomy describes the graphic elements that make up
printed letters in a typeface . Contents ( hide ) 1 Strokes 2 Terminals 3 Space 4 Proportions 5 Metal type era 6 See also 7 References 8 Further reading 9
External links Strokes ( edit ) The strokes are the components of a letterform . Strokes may be straight , as in klvwxz , or curved , as in cos . If straight ,
they may be horizontal , vertical , or diagonal ; if curved , open or closed .
</item 1>

<item 2>
Typeface anatomy
This made it possible to mix typefaces from completely different genres such as sans - serifs and serifs and have the cap height , baseline and linespacing
match perfectly , something not possible with most digital fonts . It even allowed mixing of different sizes of type with a consistent baseline . It however had
the disadvantage of often forcing typefaces to be issued with cropped descenders compared to historical typefaces , to allow tight linespacing . A `` script
line '' or `` art line '' was used for more delicate fonts with long descenders . Titling capitals , meanwhile , were issued taking up the whole space of the
metal type area , with no room for descenders . See also ( edit ) Type design Type foundry Typography Typesetting References ( edit ) Jump up ^ Studer , Anton (
29 February 2016 ) . `` Is What I See What I Get ? -- Math & Optics in Type Design '' . Typographica . Retrieved 17 April 2016 . ^ Jump up to : Carter , Bob ;
Day , Ben ; Meggs , Philip ( 2007 ) . Typographic Design : Form and Communication .
</item 2>

<item 3>
Typeface anatomy
( old style ) Didone ( modern ) Sans - serif Script Serif Slab serif Transitional Reverse - contrast Blackletter type Fraktur Rotunda Schwabacher Textualis
Gaelic type Insular Uncial Specialist Record type Punctuation Dashes Hanging punctuation Hyphen - minus Hyphenation Prime mark Quotation mark Typesetting
Calligraphy etaoin shrdlu Font computer monospaced Font catalog Letterpress Lorem ipsum Microtypography Movable type Pangram Phototypesetting Punchcutting Type
color Type design Typeface Microprint list Typographic units Agate Cicero Em En Figure space Measure Paren space Pica Point traditional point - size names Thin
space Digital typography Character encoding Font formats Hinting Rasterization Typesetting software Typographic features Web typography Related Intentionally
blank page Style guide Type foundry Commons Portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Typeface_anatomy&oldid=835865029 '' Categories :
Typography Talk Contents About Wikipedia Polski Русский Edit links This page was last edited on 11 April 2018 , at 07 : 14 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 3>

<item 4>
Typeface anatomy
Harper Design , 2013 . ^ Jump up to : Dean , Paul ( 27 March 2008 ) . `` eXtreme Type Terminology . Part 2 : Anatomy of a Letterform '' . I Love Typography .
Further reading ( edit ) Gaskell , Philip ( January 1976 ) . `` A nomenclature for the letterforms of roman type '' . Visible Language . 10 ( 1 ) : 41 -- 51 .
External links ( edit ) Wikimedia Commons has media related to Examples of typography terms . Armannsson , Sigurdur ( 6 July 2009 ) . `` Wallpaper : Font
Anatomy '' . Font.is . Full - size image ( 1920 × 1200 ) Dean , Paul ( 2008 ) . eXtreme Type Terminology -- Part 1 , Part 2 , Part 3 , Part 4 and Part 5 Devroye
, Luc . `` Hoefler & Frere - Jones '' . Retrieved 21 January 2015 . Frere - Jones , Tobias . `` Typeface Mechanics : 001 '' . Frere - Jones Type .
</item 4>

<item 5>
Typeface anatomy
ij each have a dot , jot , or tittle . Angles of strokes are called apices if at the top and vertices if at the bottom . w has one apex and two vertices ; v has
one vertex . The font shown in the example is stressed : strokes have varying widths . In this example , the stroke at the top of the g is thinner at the top
and bottom than on the sides -- a vertical stress . Terminals ( edit ) The terminals ( ends ) of instrokes and outstrokes often end in serif or sans - serif . A
seriffed terminal may be described as a wedge , bulbous , teardrop , slab , etc. , depending on the design of the type . Sorting typefaces can be practical
based on its look . A clue feature is to define the serif style . Some designs also have spurs , which are smaller than serifs and appear on angles rather than
at a terminal , as on e or G . Space ( edit ) Areas of negative space ( white space ) formed by straight or curved strokes are called counters . Closed counters
are found in abdegopq ABDOPQR , and open counters in acefhmnrstu .
</item 5>

<item 6>
Typeface anatomy
The close counter in e is also named eye . Angles of white space , as in w , are corners ( w has three corners ) ; the term is not used for angles of strokes .
The small corner formed by a serif , whether curved or angular , is called the serif bracket . Proportions ( edit ) A subtle change in proportion impacts weight
, perception , measure , and legibility . The letterform height compared to its stroke width modifies the aspect ; this slight change in weight sometimes helps
to create emphasis . The disparity between thick and thin strokes , known as stress , alters optical perception . As an example , the initial san serif
typefaces used a constantly thin stroke . The technological advances allow drawing thinner strokes . Condense type occupies less space than expanded type , then
the text written on a page can be reduced to half a page. The capline and x-height ratio improve or decrease word legibility . Metal type era ( edit ) During
the late metal type period , many fonts ( particularly in American typefounding ) were issued to `` common line '' . This meant that they were made to
standardised proportions , so that fonts of different typefaces could be mixed with no difficulty .
</item 6>

<item 7>
Typeface anatomy
Typographers also speak of an instroke , where one starts writing the letter , as at the top of acf , and an outstroke , where the pen leaves off , as at the
bottom of cejkty . Typefaces are born from the struggle between rules and results . Squeezing a square about 1 % helps it look more like a square ; to appear
the same height as a square , a circle must be measurably taller . The two strokes in an X are n't the same thickness , nor are their parallel edges actually
parallel ; the vertical stems of a lowercase alphabet are thinner than those of its capitals ; the ascender on ad is n't the same length as the descender on ap
, and so on . For the rational mind , type design can be a maddening game of drawing things differently in order to make them appear the same . Hoefler & Frere
- Jones A main vertical stroke is called a stem . The letter m has three , the left , middle , and right stems . The central stroke of an s is called the spine
.
</item 7>

<item 8>
Typeface anatomy
`` Typeface Mechanics : 001 '' . Frere - Jones Type . Retrieved 17 April 2016 . Frere - Jones , Tobias . `` Typeface Mechanics : 002 '' . Frere - Jones Type .
Retrieved 17 April 2016 . Coles , Stephen . `` The Anatomy of Type '' . Blog for Coles ' 2012 book The Anatomy of Type : A Graphic Guide to 100 Typefaces . ``
Glossary '' . FontShop . Monotype . ( hide ) Typography terminology Page Canons of page construction Column Even working Margin Page numbering Pagination Pull
quote Recto and verso Paragraph Alignment Justification Leading River Sentence spacing Widows and orphans Character Typeface anatomy Counter Diacritics Dingbat
Glyph Initial Kerning Letter - spacing Ligature Subscript and superscript Swash Text figures Tittle Capitalization ALL CAPS Camel Case Letter case Petite caps
Small caps Visual distinction Italics Oblique Bold Color Underline Blackboard bold Blackletter Infɑnt Vertical aspects Ascender Baseline Cap height Descender
Median Overshoot x-height Classifications Roman type Antiqua ( old style ) Didone ( modern ) Sans - serif Script Serif Slab serif Transitional Reverse
</item 8>

<item 9>
Typeface anatomy
Typographic Design : Form and Communication . United States of America : John Wiley & Sons . p. 31 . ISBN 9780471783909 . Jump up ^ Pecina , Martin ; Březina ,
David ( 2008 ) . `` Type Anatomy 1.0 '' . Typomil . Jump up ^ `` Introducing Ideal Sans '' . Fonts by Hoefler & Co. 4 May 2011 . Jump up ^ Bosler , Denise (
2012 - 05 - 16 ) . Mastering Type : The Essential Guide to Typography for Print and Web Design . F + W Media , Inc. p. 31 . ISBN 1440313717 . individual parts
such as the spine of the S ^ Jump up to : Dean , Paul ( 11 April 2008 ) . `` eXtreme Type Terminology . Part Three : The ' Black Art ' '' . I Love Typography .
^ Jump up to : Collen , Stephen . `` Anatomy of Type : A Graphic Guide to 100 typefaces . '' Harper Design , 2013 .
</item 9>

<item 10>
Typeface anatomy
The central stroke of an s is called the spine . When the stroke is part of a lowercase and rises above the height of an x ( called the x height ) , it is
called an ascender . Letters with ascenders are bdfhkl . A stroke which drops below the baseline is a descender . Letters with descenders are gjpqy . An arching
stroke is called a shoulder as in the top of an R or sometimes just an arch , as in hnm . A closed curved stroke is called a bowl in bdopq DOPQ ; B has two
bowls . A trailing outstroke , as in jy JQR is called a tail . The inferior diagonal stroke in K is called leg . A short horizontal stroke , as in the center of
ef and the middle stroke of EF , is called a bar . When the strokes connect as in A and H or cross strokes as in t is also known as crossbar . A longer
horizontal stroke at the top or bottom , as in ET , is called an arm . The bottom of the two - story g is called a loop ; the very short stroke at the top is
called the ear .
</item 10>
=== END ITEM id=53228d335868e46f ===

=== ITEM id=f67ed91057fe1da4 ===
Gold sentences and facts:
Sentence 0: The phrase `` objects in ( the ) mirror are closer than they appear '' is a safety warning that is required to be engraved on passenger side mirrors of motor vehicles in the USA , Canada , Nepal , India and Saudi Arabia . It is present because while these mirrors ' convexity gives them a useful field of view , it also makes objects appear smaller . Since smaller - appearing objects seem farther away than they actually are , a driver might make a maneuver such as a lane change assuming an adjacent vehicle is a safe distance behind , when in fact it is quite a bit closer . The warning serves as a reminder to the driver of this potential problem .
  Fact 0.0: The phrase 'objects in (the) mirror are closer than they appear' is a safety warning.
  Fact 0.1: The safety warning is required to be engraved on passenger side mirrors of motor vehicles.
  Fact 0.2: The engraving requirement applies in the USA, Canada, Nepal, India and Saudi Arabia.
  Fact 0.3: The warning is present on passenger side mirrors because their convexity gives them a useful field of view.
  Fact 0.4: The convexity of passenger side mirrors makes objects appear smaller.
  Fact 0.5: Smaller-appearing objects seem farther away than they actually are.
  Fact 0.6: A driver might make a maneuver such as a lane change assuming an adjacent vehicle is a safe distance behind.
  Fact 0.7: In fact, the adjacent vehicle is quite a bit closer than the driver assumes.
  Fact 0.8: The warning serves as a reminder to the driver of this potential problem.

Retrieved items:
<item 1>
Vanishing point
say D , will have the same vanishing point or converge at the same vanishing points . Mathematically , let q ≡ ( x , y , f ) be a point lying on the image plane
, where f is the focal length ( of the camera associated with the image ) , and let v ≡ ( x / h , y / h , f / h ) be the unit vector associated with q , where h
= √ x + y + f . If we consider a straight line in space S with the unit vector n ≡ ( n , n , n ) and its vanishing point v , the unit vector associated with v
is equal to n , assuming both are assumed to point towards the image plane . When the image plane is parallel to two world - coordinate axes , lines parallel to
the axis which is cut by this image plane will meet at infinity i.e. at the vanishing point . Lines parallel to the other two axes will not form vanishing
points as they are parallel to the image plane . This is one - point perspective .
</item 1>

<item 2>
Vanishing point
With the assumption that the environment consists of objects with only parallel or perpendicular sides , also called Lego - land , using vanishing points
constructed in a single image of the scene they recovered the 3D geometry of the scene . Similar ideas are also used in the field of robotics , mainly in
navigation and autonomous vehicles , and in areas concerned with object detection . See also ( edit ) Graphical projection References ( edit ) ^ Jump up to :
Kirsti Andersen ( 2007 ) Geometry of an Art , p. xxx , Springer , ISBN 0 - 387 - 25961 - 9 ^ Jump up to : B. Caprile , V. Torre ( 1 ) `` Using Vanishing Points
for Camera Calibration '' , International Journal of Computer Vision , Volume 4 , Issue 2 , pp. 127 - 139 , March 1990 Jump up ^ H. Ditton ( 1712 ) Treatise on
Perspective , p 45 Jump up ^ S.T. Barnard ' Interpreting Perspective Images `` , Artificial Intelligence 21 , 1983 , pp.
</item 2>

<item 3>
Objects in mirror are closer than they appear
Explanation from PhysLink.com . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Objects_in_mirror_are_closer_than_they_appear&oldid=838121569 ''
Categories : English phrases Mirrors Car safety Hidden categories : Wikipedia articles needing clarification from March 2016 Interlanguage link template link
number Talk About Wikipedia 한국어 Edit links This page was last edited on 25 April 2018 , at 01 : 58 . Text is available under the Creative Commons Attribution -
ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark
of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Objects in mirror are closer than they appear
Iranian film Objects in Mirror ( fa ) See also ( edit ) Blind spot ( automobile ) Blind spot monitor Driver visibility Parking sensors Rear - view mirror Mind
the gap , another safety warning used at various London Underground stations that has also become a well - known catchphrase . References ( edit ) Jump up ^ For
example , in the U.S , PART 571 Federal Motor Vehicle Safety Standards , Section 571.111 S5. 4.2 `` Each convex mirror shall have permanently and indelibly
marked at the lower edge of the mirror 's reflective surface , in letters not less than 4.8 mm nor more than 6.4 mm high the words `` Objects in Mirror Are
Closer Than They Appear . '' Jump up ^ Why does the passenger side window on my car state ' objects in mirror are closer than they appear ? ' Explanation from
PhysLink.com .
</item 4>

<item 5>
Drawing
Typically this convergence point is somewhere along the horizon , as buildings are built level with the flat surface . When multiple structures are aligned with
each other , such as buildings along a street , the horizontal tops and bottoms of the structures typically converge at a vanishing point . When both the fronts
and sides of a building are drawn , then the parallel lines forming a side converge at a second point along the horizon ( which may be off the drawing paper . )
This is a two - point perspective . Converging the vertical lines to a third point above or below the horizon then produces a three - point perspective . Depth
can also be portrayed by several techniques in addition to the perspective approach above . Objects of similar size should appear ever smaller the further they
are from the viewer . Thus the back wheel of a cart appears slightly smaller than the front wheel . Depth can be portrayed through the use of texture . As the
texture of an object gets further away it becomes more compressed and busy , taking on an entirely different character than if it was close . Depth can also be
portrayed by reducing the contrast in more distant objects , and by making their colors less saturated .
</item 5>

<item 6>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 6>

<item 7>
Vanishing point
The horizon line is a theoretical line that represents the eye level of the observer . If the object is below the horizon line , its vanishing lines angle up to
the horizon line . If the object is above , they slope down . All vanishing lines end at the horizon line . Properties of vanishing points ( edit ) 1 .
Projections of two sets of parallel lines lying in some plane π appear to converge , i.e. the vanishing point associated with that pair , on a horizon line , or
vanishing line H formed by the intersection of the image plane with the plane parallel to π and passing through the pinhole . Proof : Consider the ground plane
π , as y = c which is , for the sake of simplicity , orthogonal to the image plane . Also , consider a line L that lies in the plane π , which is defined by the
equation ax + bz = d .
</item 7>

<item 8>
Objects in mirror are closer than they appear
A reference in the 2011 film Transformers : Dark of the Moon film occurs when Dino , Bumblebee and Sideswipe are protecting Sentinel from the dreads , this
appears in a brief shot in Dino 's mirror . `` Objects In The Mirror , '' a song performed by American rapper Mac Miller on his second studio album Watching
Movies with the Sound Off `` Subjects in the mirror are more far than they appear '' the title of experimental visual project by Hungarian director / production
designer Pater Sparrow `` Objects in the Mirror , '' a 2015 song by a-ha . Objects in the mirror , a song performed by Chris Miles . ( 2016 ) Objects in mirror
are closer than they appear was a level in the videogame `` Bugs Bunny : Lost in time '' ( PlayStation 1 ) . In My Little Pony : The Movie , the sentence ``
Objects in windows may be less mountainous than they appear '' is uttered .
</item 8>

<item 9>
Objects in mirror are closer than they appear
In popular culture ( edit ) Despite its origin as a utilitarian safety warning , the phrase has become a well known catch phrase that has been used for many
other purposes . Some of them are : Objects in the Mirror Are Closer Than They Appear , the 2000 debut album of rap rock group Confrontation Camp . Objects in
the Mirror Are Closer Than They Appear , a 1993 album by Nancy Moran Objects in Mirror Are Closer Than They Appear , a 2005 album by Jozef van Wissem . Objects
in Mirror Are Closer Than They Appear , a 1995 novel by Katharine Weber . Showing an approach of a Tyrannosaurus rex in the film Jurassic Park , which was
parodied in Toy Story 2 . A Far Side cartoon by Gary Larson in which the side mirror showed a very large eyeball . A reference in Ace Ventura : Pet Detective ,
in which a pejorative term replaces `` objects . '' `` Objects in the Rear View Mirror May Appear Closer than They Are , '' a 1994 song performed by Meat Loaf
and written by Jim Steinman - the opposite of the safety warning .
</item 9>

<item 10>
Objects in mirror are closer than they appear
Objects in mirror are Closer than they appear - wikipedia Objects in mirror are Closer than they appear Jump to : navigation , search Side - view mirror with
legend Wing mirror on Korean - specification vehicle . Legend in Korean reads `` Objects in mirror are closer than they appear '' . Main article : Wing mirror
The phrase `` objects in ( the ) mirror are closer than they appear '' is a safety warning that is required to be engraved on passenger side mirrors of motor
vehicles in the USA , Canada , Nepal , India and Saudi Arabia . It is present because while these mirrors ' convexity gives them a useful field of view , it
also makes objects appear smaller . Since smaller - appearing objects seem farther away than they actually are , a driver might make a maneuver such as a lane
change assuming an adjacent vehicle is a safe distance behind , when in fact it is quite a bit closer . The warning serves as a reminder to the driver of this
potential problem . Comparison of the real over-the - shoulder view ( green arrow ) and apparent through the passenger side mirror ( sum of red arrows )
evaluation of distance between the driver 's head and the object behind .
</item 10>
=== END ITEM id=f67ed91057fe1da4 ===
