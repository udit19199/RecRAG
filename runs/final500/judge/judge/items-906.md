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

=== ITEM id=93704ee60c9ae6dc ===
Gold sentences and facts:
Sentence 0: Bruce Willis as John McClane Jai Courtney as John `` Jack '' McClane , Jr . Sebastian Koch as Yuri Komarov Yuliya Snigir as Irina Komarova Sergei Kolesnikov as Viktor Chagarin Radivoje Bukvić as Alik Mary Elizabeth Winstead as Lucy McClane
  Fact 0.0: Bruce Willis plays John McClane.
  Fact 0.1: Jai Courtney plays John 'Jack' McClane, Jr.
  Fact 0.2: Sebastian Koch plays Yuri Komarov.
  Fact 0.3: Yuliya Snigir plays Irina Komarova.
  Fact 0.4: Sergei Kolesnikov plays Viktor Chagarin.
  Fact 0.5: Radivoje Bukvić plays Alik.
  Fact 0.6: Mary Elizabeth Winstead plays Lucy McClane.

Retrieved items:
<item 1>
A Good Day to Die Hard
Retrieved 2012 - 02 - 22 . Jump up ^ `` Mary Elizabeth Winstead to Return as John McClane 's Daughter in A Good Day To Die Hard '' . Collider.com. 2012 - 08 -
07 . Retrieved 2012 - 08 - 07 . Jump up ^ http://www.movie-censorship.com/report.php?ID=175700 ^ Jump up to : `` A Good Day to Die Hard ' Finds a Couple of Male
and Female Villains '' . firstshowing.net. 2012 - 04 - 04 . Retrieved 2012 - 04 - 04 . Jump up ^ `` Cole Hauser Joins Bruce Willis In A Good Day To Die Hard ''
. deadline.com. 2012 - 05 - 08 . Retrieved 2012 - 05 - 08 . ^ Jump up to : Sneider , Jeff ( 2012 - 05 - 14 ) . `` Latest in ' Die Hard ' series adds two '' .
Variety . Retrieved 2012 - 05 - 31 .
</item 1>

<item 2>
A Good Day to Die Hard
March 21 , 2013 . External links ( edit ) Official website A Good Day to Die Hard on IMDb A Good Day to Die Hard at Box Office Mojo A Good Day to Die Hard at
Rotten Tomatoes A Good Day to Die Hard at Metacritic Die Hard Films Die Hard Die Hard 2 Die Hard with a Vengeance Live Free or Die Hard A Good Day to Die Hard
Characters John McClane Video games Die Hard Die Hard Arcade Die Hard Trilogy Die Hard Trilogy 2 : Viva Las Vegas Die Hard : Nakatomi Plaza Die Hard : Vendetta
Other Die Hard : Year One The Detective novel film Nothing Lasts Forever 58 Minutes Dynamite Cop Val Verde Roderick Thorp Films directed by John Moore Behind
Enemy Lines ( 2001 ) Flight of the Phoenix ( 2004 ) The Omen ( 2006 ) Max Payne ( 2008 ) A Good Day to Die Hard ( 2013 ) I.T.
</item 2>

<item 3>
A Good Day to Die Hard
Casting ( edit ) Bruce Willis returned as John McClane , and has expressed a desire to shoot A Good Day to Die Hard and a sixth installment in the series before
retiring the character . When casting the role of Jack McClane , the studios considered several actors , including Liam Hemsworth and James Badge Dale , before
ultimately settling on Australian actor Jai Courtney . Mary Elizabeth Winstead also appeared in the film , reprising her role as McClane 's daughter Lucy .
However all her scenes were cut from the Extended release . Sebastian Koch played the film 's primary antagonist , Yuri Komarov , while Yuliya Snigir and Cole
Hauser featured as secondary characters Irina and Collins . The cast was completed by actors Amaury Nolasco as a friend of McClane , Pasha D. Lychnikoff as a
taxi driver , and Megalyn Echikunwoke , Anne Vyalitsyna , and Ivan Kamaras in smaller roles . Filming ( edit ) Production began in Hungary in April 2012 , with
the capital Budapest standing in for Moscow .
</item 3>

<item 4>
A Good Day to Die Hard
Roman Luknár makes an apperance as Anton , a business associate of Chagarin . Ganxsta Zolee makes a brief cameo as a MRAP driver . Production ( edit )
Pre-production ( edit ) Production was formally announced in 2010 , with X-Men Origins : Wolverine and The A-Team writer Skip Woods confirmed as the film 's
screenwriter . Noam Murro was originally attached to direct the film but left production to direct the 300 prequel , 300 : Rise of an Empire . John Moore was
subsequently drafted in to replace him . The film was originally titled `` Die Hard 24 / 7 '' . The media speculated that the film would be a crossover between
the Die Hard and 24 series , with Kiefer Sutherland to reprise his role as Jack Bauer alongside John McClane . This was never confirmed by the studio , and the
film 's title was later revealed to be A Good Day to Die Hard -- with no further mention of any involvement from the 24 series -- with a release date of
February 14 , 2013 .
</item 4>

<item 5>
A Good Day to Die Hard
The main plot finds McClane travelling to Russia to get his estranged son , Jack , an undercover CIA agent , out of prison . He is soon caught in the crossfire
of a global terrorist plot . Alongside Willis , the film also stars Jai Courtney , Cole Hauser , Yuliya Snigir and Sebastian Koch as the film 's villain . Talks
of a fifth Die Hard film began before the release of the fourth installment , Live Free or Die Hard , with Willis affirming that the latter would not be the
last in the series , but pre-production did not start until September 2011 , when John Moore was officially announced as the director . Filming began in April
2012 , primarily in Budapest , Hungary . A Good Day to Die Hard premiered in Los Angeles on January 31 , 2013 , coinciding with the unveiling of a Die Hard
mural at the Fox Lot , and was released in certain East and Southeast Asian countries on February 7 and in the United States and Canada on February 13 . It is
the first Die Hard film to use Dolby Atmos Surround Mixing and the first to also be released in IMAX theaters .
</item 5>

<item 6>
A Good Day to Die Hard
A Good Day to Die Hard - wikipedia A Good Day to Die Hard Jump to : navigation , search A Good Day to Die Hard Theatrical release poster Directed by John Moore
Produced by Alex Young Wyck Godfrey Written by Skip Woods Based on Characters by Roderick Thorp Starring Bruce Willis Jai Courtney Sebastian Koch Rasha Bukvić
Cole Hauser Yuliya Snigir Music by Marco Beltrami Cinematography Jonathan Sela Edited by Dan Zimmerman Production company Giant Pictures TSG Entertainment
Distributed by 20th Century Fox Release date January 31 , 2013 ( 2013 - 01 - 31 ) ( Leicester Square ) February 14 , 2013 ( 2013 - 02 - 14 ) ( United States )
Running time 97 minutes Country United States Language English Russian Budget $92 million Box office $304.7 million A Good Day to Die Hard is a 2013 American
action thriller film and the fifth installment in the Die Hard film series . The film was directed by John Moore and written by Skip Woods , and stars Bruce
Willis as John McClane .
</item 6>

<item 7>
A Good Day to Die Hard
Komarov remarks that Jack will get to watch his father die , prompting Jack to hurl him off the rooftop into the path of the spinning helicopter rotors ,
killing him . As Jack and John reunite inside the building , Irina tries to avenge her father by ramming the helicopter , now out of ammunition , into the
building in a suicide attack . Father and son survive by leaping off the building into a large pool of water , as the helicopter crashes and explodes , killing
Irina . In the end , the McClanes return home and reunite with Lucy on the tarmac . Cast ( edit ) Bruce Willis as John McClane Jai Courtney as John `` Jack ''
McClane , Jr . Sebastian Koch as Yuri Komarov Yuliya Snigir as Irina Komarova Sergei Kolesnikov as Viktor Chagarin Radivoje Bukvić as Alik Mary Elizabeth
Winstead as Lucy McClane Additional cast members include CIA operatives : Aldis Hodge as Lt. Foxy and Cole Hauser as Mike Collins . Amaury Nolasco appears as
Murphy , an NYPD detective and McClane 's friend .
</item 7>

<item 8>
A Good Day to Die Hard
Rick Groen of The Globe and Mail criticized the action scenes as being `` messy '' , but concluded his review saying that the film `` continues the franchise
without undue embarrassment . '' Peter Howell of The Toronto Star remarked that Willis and Courtney made a strong estranged family duo and that the film had a
nice drinking game routine going for it with how many times McClane exclaims `` I 'm on vacation ! '' during the running time . Box Office ( edit ) A Good Day
to Die Hard grossed $67,349,198 in North America and $237,304,984 in other territories for a worldwide total of $304,654,182 , collecting more than three times
its $92 million budget . In North America , A Good Day to Die Hard grossed an estimated $840,000 from its night showings at 2,328 locations on Wednesday ,
February 13 , 2013 . The next day , at an additional 1,225 locations , the film managed to accumulate $8,239,116 , opening at # 2 behind Safe Haven .
</item 8>

<item 9>
A Good Day to Die Hard
A Good Day to Die Hard was a box office success but received overwhelmingly negative reviews for its implausible action sequences , cinematography , weak plot ,
cliched script , and lack of characterization , although the special effects were praised . It is the first film produced by TSG Entertainment , since
distributor 20th Century Fox 's departure from Dune Entertainment upon the completion of their distribution contract at the end of 2012 . Contents ( hide ) 1
Plot 2 Cast 3 Production 3.1 Pre-production 3.2 Casting 3.3 Filming 3.4 Post-production 4 Soundtrack 5 Release 6 Reception 6.1 Box office 6.2 Home media 7
References 8 External links Plot ( edit ) In Moscow , Viktor Chagarin , a high - ranking , but corrupt , Russian official plans to incriminate former
billionaire and government whistleblower Yuri Komarov in an imminent rigged trial unless Komarov hands over a secret file believed to contain evidence
incriminating Chagarin . Separately , Jack McClane , who has been arrested after an assassination attempt , negotiates for a shorter sentence by offering to
testify against Komarov .
</item 9>

<item 10>
A Good Day to Die Hard
`` Get to the Choppa ! '' 2 : 59 23 . `` Chopper Takedown '' 3 : 26 24 . `` It 's Hard to Kill a McClane '' 2 : 59 25 . `` Triple Vodka Rhapsody '' 1 : 55 26 .
`` McClane 's Brain '' 2 : 00 Release ( edit ) # 46 TruSpeed Porsche GT3 during a race of the 2012 Supercup ( that ran as a support to the 2012 USGP ) promoting
the release date on February 14 , 2013 . On January 31 , 2013 , 20th Century Fox held a special tribute to the 25th anniversary of the Die Hard series by
unveiling a mural of a scene from Die Hard ( 1988 ) on Sound Stage 8 of the Fox Lot . Afterward , the premiere screening of A Good Day to Die Hard was shown .
Two additional premieres were held in Europe leading up to the film 's commercial release : one in Berlin , Germany on February 4 and the second in London ,
England on February 7 .
</item 10>
=== END ITEM id=93704ee60c9ae6dc ===

=== ITEM id=fd1b9a6319f3dd8f ===
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
The Grand Tour (TV series)
ASIN B0731ZQ3R7. 1 . Past , present or future Missing or empty url = ( help ) ; access - date = requires url = ( help ) ^ Jump up to : Starkey , Adam ( 13
December 2017 ) . `` The Grand Tour season 2 : Richard Hammond is on crutches in episode 2 after horrifying crash '' . Metro . London . Retrieved 13 December
2017 . second episode titled The Falls Guys , Clarkson drives ... Ford GT from New York City to Niagara Falls , with James May making the same journey via
public transport with Hammond ... Celebrity Face Off ... cricketer Kevin Pietersen and baseball 's Brian Wilson Jump up ^ `` The Grand Tour Season 2 Episode 3
'' . 16 December 2017 . Retrieved 16 December 2017 . ^ Jump up to : `` Zadarska manekenka Silvija oduševila u ' The Grand Touru ' '' ( Zadar model Silvija
delights in The Grand Tour ) . 24sata ( in Croatian ) . 3 January 2018 .
</item 1>

<item 2>
The Grand Tour (TV series)
The Sunday Times : Driving . Retrieved 14 December 2017 . The show has a new test track in Enstone , ... there planning problems when ( Clarkson ) tried once
before to use the Enstone track , back in Top Gear days ... The noisier testing of fast cars will continue to be done in Wiltshire at RAF Wroughton ...
nicknamed `` the Eboladrome '' Jump up ^ Reed , Jason ( 24 January 2018 ) . `` The Secret Story Behind The Grand Tour 's Celebrity Braincrash '' . Shifting
Lanes . Retrieved 26 January 2018 . ^ Jump up to : McCreesh , Louise ( 27 November 2017 ) . `` The Grand Tour season 2 confirms celebrity guests including
Kiefer Sutherland and David Hasselhoff '' . Retrieved 19 December 2017 . The celebrity guests this series include Luke Evans , Kiefer Sutherland , Hugh
Bonneville , Kevin Pietersen and Dominic Cooper ( ... ) Dynamo , Rory Mcllroy , Michael Ball and Alfie Boe will also appear in season two .
</item 2>

<item 3>
The Grand Tour (TV series)
For episode 2 , the list included 5.11 Tactical . A DHL Boeing 757 was featured in the opening sequence of episode 5 , the tent was located in Rotterdam , and
the DHL logo is featured on part of the crash barrier at the Eboladrome . Technical issues ( edit ) During filming for some episodes in the second series of the
show , the production team suffered major technical issues with the tent lighting , which threatened to derail filming . Nonetheless , all episodes were
released on the correct dates , with all delays having been averted . Reception ( edit ) As of November 2016 the show has received positive reviews from critics
, with The Guardian saying `` Jeremy Clarkson and co leave the BBC in their dust '' . Daily Express TV reporter , Neela Debnath commented that the first episode
`` resembled a Hollywood blockbuster '' and added that `` ( The Grand Tour is ) basically Top Gear on steroids '' . However , BBC Arts Editor , Will Gompertz
said of the opening that `` there is no irony .
</item 3>

<item 4>
The Grand Tour (TV series)
The other two are so intoxicated by their own debate they do n't notice . 2.10 It 's time to step in a dog turd of chat , on Conversation Street . The stools
have disappeared . Clarkson with a sledge hammer in his hands , Hammond with a shovel are standing above a third person ( May ? ) They are swinging their tools
at the prone body which cries out in pain when it is hit . ( this is the only time in two series that we hear any one of the three make a noise in an otherwise
silent introduction ) Running gags ( edit ) The first series included several running gags . During the opening titles of each episode , a camera drone was
destroyed . From episode three onward , one of the presenters ' names is always seen misspelt during the opening . For `` Celebrity Brain Crash '' , celebrity
guests are humorously `` killed '' in an accident while making their way to the tent . After each `` death '' , May asks `` Does that mean he / she 's not
coming on then ? ''
</item 4>

<item 5>
The Grand Tour (TV series)
1 Wet not shown Volkswagen Up ! GTI 1 : 39.7 Wet 18 Enstone ( edit ) Gravel course at Enstone Airfield introduced in Series 2 and used for Celebrity Face Off
For the second series , a new test track was created at Enstone Airfield , close to the fixed studio tent location , and which the production team had
previously tried to use for Top Gear . The main Eboladrome track was planned to continue to serve for the testing of loud and fast cars . Celebrity Face off (
edit ) Following complaints regarding the `` Celebrity Brain Crash '' segment in the first series , this segment was replaced by `` Celebrity Face Off '' . Two
celebrities join the show each week , with a similar background or connection and often from different countries , to see which one of the two is faster . The
car used for this segment is a Jaguar F - Type R - Dynamic coupe , racing around a half gravel , half asphalt track at Enstone . Prior to showing the lap
results , Clarkson interviews the guests . Lap board Celebrity Time Condition Episode Ricky Wilson 1 : 20.1 Dry 14 David Hasselhoff 1 : 24.
</item 5>

<item 6>
The Grand Tour (TV series)
The Daily Telegraph , while not as positive , still approved of the episode , stating `` The writing is still rather ropey . Clarkson 's suggestion of a new
nickname for May -- `` Dingleberry Handpump '' -- failed to raise a titter even among the super-fans gathered for the London premiere '' but also said that ``
for each wobble , there are just as many moments when The Grand Tour manages the clever trick Top Gear could pull off at its best : raising a chuckle while
sneaking in a bit of serious journalism at the same time . '' and ultimately gave the episode 3 out of 5 stars . Jeremy Clarkson himself believed that they had
`` hit the ground running with series 2 of the Grand Tour '' . Radio Times has criticised the gender balance of guests on the `` Celebrity Face Off '' segment .
There was only 1 woman guest in the episode `` Oh Canada '' ; Paris Hilton . Hothouse Flowers playing for the opening sequence of `` The Holy Trinity '' episode
in the Lucerne Valley , California Broadcast ( edit ) Australian free - to - air network Seven Network started broadcasting the first series of The Grand Tour
in mid-October 2017 .
</item 6>

<item 7>
The Grand Tour (TV series)
Rotten Tomatoes . Retrieved 17 July 2017 . Jump up ^ `` The Grand Tour season 2 episode 1 review : We 're in safe hands as Clarkson , Hammond and May return ''
. Digital Spy . Retrieved 8 December 2017 . Jump up ^ `` Television review : The Grand Tour on Amazon Prime '' . The Times . Retrieved 8 December 2017 . Jump up
^ `` The Grand Tour season 2 , episode 1 review : Has Jeremy Clarkson finally found his groove ? '' . The Daily Telegraph . Retrieved 8 December 2017 . Jump up
^ Clarkson , Jeremy ( 8 December 2017 ) . `` Jeremy Clarkson on Twitter '' . twitter.com . Retrieved 15 December 2017 . Jump up ^ Ling , Thomas ( 27 November
2017 ) . `` The Grand Tour 's first celebrity guests for series 2 have been revealed -- and there 's not a single woman '' . Retrieved 20 December 2017 . Jump
up ^ Long , Trevor .
</item 7>

<item 8>
The Grand Tour (TV series)
The Sun . Retrieved 23 December 2017 . ^ Jump up to : Daly , Emma ( 26 November 2016 ) . `` Jeremy Clarkson forced to ' kill off ' celebs on The Grand Tour
after BBC ' bans Top Gear rip - off star interviews ' '' . The Sun . Retrieved 26 November 2016 . Jump up ^ Lee , Kristen ( 16 November 2017 ) . `` ' Celebrity
Brain Crash ' Axed on the Grand Tour Season 2 '' . jalopnik.com . Retrieved 17 November 2017 . Jump up ^ McCreesh , Louise ( 27 November 2017 ) . `` The Grand
Tour season 2 confirms celebrity guests including The Hoff , Jack Bauer and Luke Evans '' . Digital Spy . Retrieved 28 November 2017 . Jump up ^ Hsieh , Tony (
20 November 2016 ) . `` The Grand Tour : 3 Mustangs from the Opening Sequence of the Premiere Episode '' . Grand Tour Nation . Retrieved 24 November 2016 . ^
Jump up to : `` Amazon Instant Video '' .
</item 8>

<item 9>
The Grand Tour (TV series)
1 Notes 2.3 Series 2 ( 2017 -- 18 ) 2.3. 1 Notes 2.4 Series 3 3 Production 3.1 History 3.2 Legal issues 3.3 Filming 3.4 Promotion 3.5 Sponsorship 3.6 Technical
Issues 4 Reception 5 Broadcast 6 See also 7 References 8 External links Format ( edit ) The Grand Tour is presented as a mix of pre-recorded television films
and live - audience segments presented in a large tent to an audience of about 300 members . The show and its segments are hosted by Jeremy Clarkson , Richard
Hammond , and James May . The show is presented from a pair of large tents on a Grand Tour around the world . The presenters sit around a trestle table , with
the studio audience seated in front of them . Films ( edit ) Each show is generally bookended by two television films , or two parts of the same film , produced
by one or more of the hosts .
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
=== END ITEM id=fd1b9a6319f3dd8f ===

=== ITEM id=12412b6baf848a4a ===
Gold sentences and facts:
Sentence 0: The Quebec Winter Carnival or the Carnaval de Québec is a pre-Lenten festival held in Quebec City . After being held intermittently since 1894 , the Carnaval de Québec has been celebrated annually since 1955 . That year Bonhomme Carnaval , the mascot of the festival , made his first appearance . Up to one million people attended the Carnaval de Québec in 2006 making it the largest winter festival in the world .
  Fact 0.0: The Quebec Winter Carnival (Carnaval de Québec) is a pre-Lenten festival.
  Fact 0.1: The Carnaval de Québec is held in Quebec City.
  Fact 0.2: The Carnaval de Québec was held intermittently after 1894.
  Fact 0.3: The Carnaval de Québec has been celebrated annually since 1955.
  Fact 0.4: In 1955, Bonhomme Carnaval, the mascot of the festival, made his first appearance.
  Fact 0.5: Up to one million people attended the Carnaval de Québec in 2006.
  Fact 0.6: In 2006, the Carnaval de Québec was the largest winter festival in the world.

Retrieved items:
<item 1>
God Save the Queen
Queen 's Printer for Canada . Retrieved 5 July 2010 . Jump up ^ Remembrance Day ( PDF ) , Vancouver Board of Parks and Recreation , 11 November 2009 , archived
from the original ( PDF ) on 7 July 2011 , retrieved 5 July 2010 Jump up ^ `` Honours and Salutes '' . Queen 's Printer for Canada . Retrieved 4 March 2017 . ^
Jump up to : Bélanger , Claude . `` The Quebec History Encyclopedia '' . In Marianopolis College . National Anthem of Canada . Montreal : Marianopolis College .
Retrieved 5 July 2010 . ^ Jump up to : Department of Canadian Heritage . `` National Anthem : O Canada > Parliamentary Action '' . Queen 's Printer for Canada .
Retrieved 29 June 2010 . ^ Jump up to : Department of National Defence ( 1 April 1999 ) , The Honours , Flags and Heritage Structure of the Canadian Forces (
PDF ) , Ottawa : Queen 's Printer for Canada , p.
</item 1>

<item 2>
Quebec Winter Carnival
About Wikipedia
</item 2>

<item 3>
Quebec Winter Carnival
Frommer 's Canada : With the best hiking & outdoor adventures . John Wiley & Sons . p. 300 . ISBN 978 - 0 - 470 - 04457 - 5 . Jump up ^
https://www.theglobeandmail.com/life/travel/activities-and-interests/the-snowman-behind-the-legend-get-to-know-quebecs-beloved-bonhomme/article37716316/ Quebec
Winter Carnaval virtual tour in panoramic photography Tornoipee-wee.qc.ca Fissnowboard.com Icehotel-canada.com Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Quebec_Winter_Carnival&oldid=822352977 '' Categories : Tourist attractions in Quebec City Festivals in Quebec City
Winter festivals in Canada Mardi Gras 1894 establishments in Quebec Hidden categories : Articles needing additional references from January 2018 All articles
needing additional references Quebec articles missing geocoordinate data All articles needing coordinates Talk Contents About Wikipedia Deutsch Español Français
한국어 日本 語 Português Svenska Edit links This page was last edited on 25 January 2018 , at 21 : 54 .
</item 3>

<item 4>
Quebec Winter Carnival
Bonhomme is described as a seven - foot - tall , four - hundred pound snowman sporting a red cap , black buttons and a ceinture fléchée . It is traditional to
drink Caribou , a hot alcoholic beverage , to keep warm . The public auction is a fundraising event in aid of the carnival . This auction features many goods
and services donated for silent auction and live auction . Feasts and restaurants ( edit ) The Business Leaders ' Luncheon , organized by the Québec City
Chamber of Commerce . The Calgary Stampede Flap Jack Breakfast . In 2008 , carnival - goers were served by Prime Minister Stephen Harper . The ' Restaurant
Partners ' Campaign is a 179 - day promotion during which Québec City restaurants offer customers a special menu for a fixed price throughout the carnival (
including appetizer , soup , or salad , a main course , and a dessert ) . Races and tournaments ( edit ) A sleigh race in which drivers and their horses take
part in a single - and double - harness race on the Plains of Abraham . An Ice canoe race on the St. Lawrence river .
</item 4>

<item 5>
Quebec Winter Carnival
Quebec winter carnival - wikipedia Quebec winter carnival Jump to : navigation , search This article needs additional citations for verification . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2018 ) ( Learn how and when to
remove this template message ) Quebec Winter Carnival Carnaval de Québec Bonhomme Carnaval in 2011 Genre Festival Begins End of January Ends Mid February
Frequency Annually Location ( s ) Quebec City Country Canada Website Carnaval de Québec The Quebec Winter Carnival or the Carnaval de Québec is a pre-Lenten
festival held in Quebec City . After being held intermittently since 1894 , the Carnaval de Québec has been celebrated annually since 1955 . That year Bonhomme
Carnaval , the mascot of the festival , made his first appearance . Up to one million people attended the Carnaval de Québec in 2006 making it the largest
winter festival in the world .
</item 5>

<item 6>
Quebec Winter Carnival
Free outdoor public banquets ( brunch , breakfast , etc . ) . . The Canadian , Québécois , International and Student artist snow sculpture contests on the
Plains of Abraham , the main setting of the carnival . The Plains are a public city park and stay open for leisure activities , including snowshoeing and
cross-country skiing trails , during carnival time . Part of the Plains around the Citadel is transformed into an outdoor winter amusement park with various
family - themed activities , including the display of the three main snow - sculpting contests ( Canada 's provinces , Quebec 's regions , International ) and
the traditional bikini snow bath event ( bain de neige ) Outdoor dance parties are held at the Ice Palaces . Kiosks and other outlets in the city sell the
Bonhomme effigy tag that grants admission into most of the events , although some are free outside the main site . Most commercial main streets are decorated
and some bars and restaurants set up a winter patio in front of their establishments . Bonhomme -- short for bonhomme de neige ( `` snowman '' ) is the official
ambassador of the festivities , the castle lord of the Ice Palace .
</item 6>

<item 7>
2018 Winter Olympics
Jump up ^ `` Pyeongchang 2018 Winter Olympics '' . worldcurling.org . Retrieved 23 July 2016 . Jump up ^ `` ISU Communication no . 2136 '' . International
Skating Union . Retrieved 31 December 2017 . Jump up ^ `` PyeongChang 2018 Launches Official Emblem '' . olympic.org . International Olympic Committee .
Retrieved 12 August 2016 . Jump up ^ `` PyeongChang 2018 Pictograms '' . The PyeongChang 2018 Olympic and Paralympic Winter Games . 25 January 2017 . Retrieved
30 May 2017 . Jump up ^ `` Tiger and bear mascots unveiled for Pyeongchang 2018 Winter Olympics and Paralympics '' . insidethegames.biz . Dunsar Media Company
Limited . 2 June 2016 . Retrieved 16 November 2017 .
</item 7>

<item 8>
Quebec Winter Carnival
Contents ( hide ) 1 Activities and attractions 2 Feasts and restaurants 3 Races and tournaments 4 Gallery 5 References Activities and attractions ( edit ) The
most famous attractions of this winter festival are the night - time and daytime parades led by mascot Bonhomme Carnaval . The parades wind through the upper
city , decorated for the occasion with lights and ice sculptures . Numerous public and private parties , shows and balls are held across the city , some of them
outside in the bitter cold , testimony to the Québécois ' fabled joie de vivre . Other major events include : A masquerade ball with up to 400 participants at
the grand ballroom of the Château Frontenac . The opening and closing ceremonies taking place at the Ice Palace before thousands of participants , Bonhomme and
the mayor of Quebec . Outdoor sport events ( snowboarding , ice canoe , snowshoes , hockey , dog - sledding , etc. , some of them part of World Championship
tournaments ) inside and outside the city . Free outdoor public banquets ( brunch , breakfast , etc . ) . .
</item 8>

<item 9>
Quebec Winter Carnival
An Ice canoe race on the St. Lawrence river . The Québec International Pee - Wee Hockey Tournament ( not officially part of the carnival program since 1977 ) .
The Snowboard World Cup in Quebec City ( not part of the carnival official program ) . Also not part of the official program but worth a visit or a stay , the
Quebec City Ice Hotel is open every year from early January to late March with its bar , nightclub , exhibition galleries and ceremonial chapel . Gallery ( edit
) Ice sculpture ( 2005 ) Family venue ( 2006 ) Ice Palace ( 2009 ) Parade in Montreal ( 2011 ) Canoe race ( 2011 ) Ice Palace ( 2011 ) Toboggan at Terrasse
Dufferin References ( edit ) Jump up ^ Daniel Shafto ( 1 January 2009 ) . Carnival . Infobase Publishing . p. 80 . ISBN 978 - 1 - 4381 - 2660 - 9 . Jump up ^
Hilary Davidson ; Paul Karr ; Herbert Bailey Livesey ; Bill McRae ; Donald Olson ( 14 August 2006 ) .
</item 9>

<item 10>
Graph query row:
entity: Bonhomme Carnaval
relationship: RELATED_TO
role: mascot
carnival: Quebec Winter Carnival
</item 10>
=== END ITEM id=12412b6baf848a4a ===

=== ITEM id=440aeb4174353054 ===
Gold sentences and facts:
Sentence 0: Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi. tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16 urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China .
  Fact 0.0: Beijing is the capital of the People's Republic of China.
  Fact 0.1: Beijing was formerly romanized as Peking.
  Fact 0.2: Beijing is the world's second most populous city proper.
  Fact 0.3: Beijing is the most populous capital city.
  Fact 0.4: Beijing is located in northern China.
  Fact 0.5: Beijing is governed as a direct-controlled municipality under the national government.
  Fact 0.6: Beijing has 16 urban, suburban, and rural districts.
  Fact 0.7: Beijing Municipality is surrounded by Hebei Province, except for the neighboring Tianjin Municipality to the southeast.
  Fact 0.8: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the Jingjinji metropolitan region.
  Fact 0.9: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the national capital region of China.
Sentence 1: Over the past 3,000 years , the city of Beijing has had numerous other names . The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University , still use the former romanization .
  Fact 1.0: Over the past 3,000 years, the city of Beijing has had numerous other names.
  Fact 1.1: The name Beijing means 'Northern Capital'.
  Fact 1.2: The name Beijing is made of the Chinese characters 北 (north) and 京 (capital).
  Fact 1.3: The name Beijing was applied to the city in 1403 during the Ming dynasty.
  Fact 1.4: The name Beijing was applied to distinguish the city from Nanjing, the 'Southern Capital'.
  Fact 1.5: The English spelling Beijing is based on the pinyin romanization of the two characters as pronounced in Standard Mandarin.
  Fact 1.6: The older English spelling Peking is the postal romanization of the same two characters as pronounced in Chinese dialects spoken in the southern port towns.
  Fact 1.7: The southern port towns were the first visited by European traders and missionaries.
  Fact 1.8: The Chinese dialects spoken in the southern port towns preserve the Middle Chinese pronunciation of 京 as kjaeng.
  Fact 1.9: That pronunciation of 京 as kjaeng comes before a phonetic shift in the northern dialects to the modern pronunciation.
  Fact 1.10: Peking is no longer the common name for the city.
  Fact 1.11: Some of the city's older locations and facilities still use the former romanization Peking.
  Fact 1.12: Beijing Capital International Airport, with IATA code PEK, still uses the former romanization.
  Fact 1.13: Peking University still uses the former romanization.

Retrieved items:
<item 1>
Beijing
Two generations later , Kublai Khan ordered the construction of Dadu ( or Daidu to the Mongols , commonly known as Khanbaliq ) , a new capital for his Yuan
dynasty to the northeast of the Zhongdu ruins . The construction took from 1264 to 1293 , but greatly enhanced the status of a city on the northern fringe of
China proper . The city was centered on the Drum Tower slightly to the north of modern Beijing and stretched from the present - day Chang'an Avenue to the
northern part of Line 10 subway . Remnants of the Yuan rammed earth wall still stand and are known as the Tucheng . Ming dynasty One of the corner towers of the
Forbidden City , which was built by Emperor Yongle in Ming dynasty . In 1368 , soon after declaring the new Hongwu era of the Ming dynasty , the rebel leader
Zhu Yuanzhang sent an army to Khanbaliq and conquered it . Since the Yuan continued to occupy Shangdu and Mongolia , however , Dadu was renamed to Beiping as
used to supply the military garrisons in the area .
</item 1>

<item 2>
Beijing
The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the
Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two
characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are
pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese
pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for
the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University ,
still use the former romanization . The single Chinese character abbreviation for Beijing is 京 , which appears on automobile license plates in the city . The
official Latin alphabet abbreviation for Beijing is `` BJ '' .
</item 2>

<item 3>
Beijing
tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and
most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16
urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the
southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China . As a city combining both modern and
traditional architecture , Beijing is a megacity rich in history , exemplified in its global influence in politics , economy , education , history , culture ,
and technology . Beijing is the second largest Chinese city by urban population after Shanghai and is the nation 's political , cultural , and educational
center . It is home to the headquarters of most of China 's largest state - owned companies and is a major hub for the national highway , expressway , railway ,
and high - speed rail networks .
</item 3>

<item 4>
Beijing
Retrieved 22 July 2009 . External links Find more aboutBeijingat Wikipedia 's sister projects Definitions from Wiktionary Media from Wikimedia Commons News from
Wikinews Quotations from Wikiquote Texts from Wikisource Textbooks from Wikibooks Travel guide from Wikivoyage Learning resources from Wikiversity Economic
profile for Beijing at HKTDC Visit Beijing Facebook Page Photograph of The approach to Peking -- outside the walls taken in 1890 by Sir Henry Norman Preceded by
Lin'an ( Song dynasty ) Capital of China ( as Dadu of Yuan ) 1264 -- 1368 Succeeded by Nanjing ( Ming dynasty ) Preceded by Nanjing ( Ming dynasty ) Capital of
China 1420 -- 1928 Succeeded by Nanjing ( ROC ) Preceded by Nanjing ( ROC ) Capital of the People 's Republic of China 1949 -- present Succeeded by present
capital Places adjacent to Beijing Zhangjiakou , Hebei Chengde , Hebei Chengde , Hebei Zhangjiakou , Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang ,
</item 4>

<item 5>
Beijing
The official Latin alphabet abbreviation for Beijing is `` BJ '' . History Main article : History of Beijing Early History The earliest traces of human
habitation in the Beijing municipality were found in the caves of Dragon Bone Hill near the village of Zhoukoudian in Fangshan District , where Peking Man lived
. Homo erectus fossils from the caves date to 230,000 to 250,000 years ago . Paleolithic Homo sapiens also lived there more recently , about 27,000 years ago .
Archaeologists have found neolithic settlements throughout the municipality , including in Wangfujing , located in downtown Beijing . The first walled city in
Beijing was Jicheng , the capital city of the state of Ji and was built in 1045 BC . Within modern Beijing , Jicheng was located around the present Guang'anmen
area in the south of Xicheng District . This settlement was later conquered by the state of Yan and made its capital . Early Imperial China The Tianning Pagoda
, built around 1120 during the Liao dynasty . After the First Emperor unified China , Jicheng became a prefectural capital for the region .
</item 5>

<item 6>
Beijing
1 Towns 4.1. 2 Neighborhoods 4.2 Judiciary and procuracy 4.3 Diplomatic missions 5 Economy 5.1 Sector composition 5.2 Economic zones 6 Demographics 6.1
Metropolitan area 7 Culture 7.1 Places of interest 8 Religion 8.1 Chinese folk religion and Taoism 8.2 Buddhism 8.3 Islam 8.4 Christianity 9 Media 9.1
Television and radio 9.2 Press 10 Sports 10.1 Events 10.2 Venues 10.3 Clubs 11 Transportation 11.1 Rail and high - speed rail 11.2 Roads and expressways 11.3
Air 11.4 Public transit 11.5 Taxi 11.6 Bicycles 12 Defense and aerospace 13 Nature and wildlife 14 See also 15 References 16 Further reading 17 External links
Etymology See also : Names of Beijing Over the past 3,000 years , the city of Beijing has had numerous other names .
</item 6>

<item 7>
Beijing
In 938 , after the fall of the Tang , the Later Jin ceded the entire northern frontier to the Khitan Liao dynasty , which treated the city as Nanjing , or the
`` Southern Capital '' , one of four secondary capitals to complement its `` Supreme Capital '' , Shangjing ( modern Baarin Left Banner in Inner Mongolia ) .
Some of the oldest surviving structures in Beijing date to the Liao period , including the Tianning Pagoda . Miaoying Temple and its White Stupa in Dadu which
was built by Kublai Khan The Liao fell to the Jurchen Jin dynasty in 1122 , which gave the city to the Song dynasty and then retook it in 1125 during its
conquest of northern China . In 1153 , the Jurchen Jin made Beijing their `` Central Capital '' , or Zhongdu . The city was besieged by Genghis Khan 's invading
Mongolian army in 1213 and razed to the ground two years later .
</item 7>

<item 8>
Beijing
Beijing - wikipedia Beijing Jump to : navigation , search `` Peking '' redirects here . For other uses , see Beijing ( disambiguation ) and Peking (
disambiguation ) . Beijing 北京 市 Municipality Beijing Municipality Clockwise from top : Beijing CBD skyline , Tiananmen , Temple of Heaven , National Center for
the Performing Arts , and Beijing National Stadium Location of Beijing Municipality within China Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383
° E ﻿ / 39.917 ; 116.383 Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383 ° E ﻿ / 39.917 ; 116.383 Country China Established date 1045 BCE ( Zhou
Dynasty ) Divisions - County - level - Township - level 16 districts 289 towns and villages Government Type Municipality Party Secretary Cai Qi Mayor Chen
Jining Congress Chairman Li Wei ( zh ) Conference Chairman Ji Lin Area Municipality 16,411 km ( 6,336 sq mi ) Land 16,
</item 8>

<item 9>
Beijing
1928 -- 1937 ; 1945 -- 1949 ) Literal meaning `` Northern Capital '' ( show ) Transcriptions Standard Mandarin Hanyu Pinyin Běijīng Bopomofo ㄅ ㄟ ˇ ㄐ ㄧ ㄥ Gwoyeu
Romatzyh Beeijing Wade -- Giles Pei - ching IPA ( pèi. tɕíŋ ) Wu Suzhounese Poh - cin Hakka Romanization Bet - gin Yue : Cantonese Yale Romanization Bākgìng or
Bākgīng IPA ( pɐ́k̚. kêŋ ) or ( pɐ́k̚. kéŋ ) Jyutping Bak1ging1 Southern Min Hokkien POJ Pak - kian Tâi - lô Pak - kiann Eastern Min Fuzhou BUC Báe̤k - gĭng
Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi.
</item 9>

<item 10>
Beijing
And under the Hongwu Emperor 's feudal policies it was given to Zhu Di , one of his sons , who was created `` Prince of Yan '' . The early death of Zhu
Yuanzhang 's heir led to a succession struggle on his death , one that ended with the victory of Zhu Di and the declaration of the new Yongle era . Since his
harsh treatment of the Ming capital Yingtian ( modern Nanjing ) alienated many there , he established his fief as a new co-capital . The city of Beiping became
Beijing ( northern capital ) or Shuntian in 1403 . The construction of the new imperial residence , the Forbidden City , took from 1406 to 1420 ; this period
was also responsible for several other of the modern city 's major attractions , such as the Temple of Heaven and Tian'anmen . On 28 October 1420 , the city was
officially designated the capital of the Ming dynasty in the same year that the Forbidden City was completed . Beijing became the empire 's primary capital and
Yingtian , also called Nanjing ( southern capital ) , became the co-capital .
</item 10>
=== END ITEM id=440aeb4174353054 ===

=== ITEM id=a5f8f3ed05c9a1d4 ===
Gold sentences and facts:
Sentence 0: Subjects are all taken on the same level - there is no higher or standard grade as in the past . There are three pass levels , each with different pass requirements. First is a pass with Higher certificate , one must obtain 40 % in home language , 40 % in two other subjects and must obtain 30 % in three other subjects . A learner who passed matric with higher certificate level , can not enroll for a University Degree nor a Diploma at an institute of higher learning . The mean mark in any subject is usually about 55 . Only a small proportion of candidates score an ' A ' in any subject ( from as little as 2 % to a maximum of about 10 % in subjects taken by highly select groups . ) A further 8 -- 15 % are likely to gain a ' B ' and about 20 -- 25 % achieve a ' C ' grade . The National Senior Certificate is a group certificate and records an aggregate mark .
  Fact 0.0: All subjects are taken on the same level, and there is no higher or standard grade as in the past.
  Fact 0.1: There are three pass levels, each with different pass requirements.
  Fact 0.2: The first pass level is a pass with Higher certificate.
  Fact 0.3: To pass with Higher certificate, a learner must obtain 40% in home language, 40% in two other subjects, and 30% in three other subjects.
  Fact 0.4: A learner who passed matric with Higher certificate level cannot enroll for a University Degree.
  Fact 0.5: A learner who passed matric with Higher certificate level cannot enroll for a Diploma at an institute of higher learning.
  Fact 0.6: The mean mark in any subject is usually about 55.
  Fact 0.7: Only a small proportion of candidates score an 'A' in any subject, from as little as 2% to a maximum of about 10% in subjects taken by highly select groups.
  Fact 0.8: A further 8 to 15% of candidates are likely to gain a 'B' in a subject.
  Fact 0.9: About 20 to 25% of candidates achieve a 'C' grade.
  Fact 0.10: The National Senior Certificate is a group certificate.
  Fact 0.11: The National Senior Certificate records an aggregate mark.

Retrieved items:
<item 1>
National Senior Certificate
The IEB operates on a national level catering primarily for independent schools . Contents ( hide ) 1 Subject requirements 1.1 Group A : Fundamentals (
Compulsory ) 1.2 Group B : Electives 2 Life Orientation 3 Assessment 4 Level system ( 1 to 7 ) 5 The 5 % language compensation 6 University entrance 7 See also
8 References 9 External links Subject requirements. ( edit ) Learners study at least 7 subjects - 4 compulsory and at least 3 electives . All subjects are
written on one grade only and are no longer written on Higher or Standard Grade . Not all schools offer the full range of Elective subjects listed here . Each
school may offer subjects specific to its academic orientation . For example , Agriculture Schools offer the agriculture - orientated subjects whereas technical
Schools offer the practical and mechanical - orientated subjects.
</item 1>

<item 2>
National Senior Certificate
V Group a : fundamentals ( compulsory ) ( edit ) Learners must study 4 compulsory subjects , Two official languages of South Africa must be studied : Afrikaans
English Ndebele Northern Sotho Sesotho Siswati Tsonga Tswana Venda Xhosa Zulu Mathematics or Mathematical Literacy Life Orientation Group B : electives ( edit )
At least 3 subjects from the following : Accounting Agricultural Management Practices Agricultural Sciences Agricultural Technology Business Studies Civil
Technology Computer Applications Technology ( previously Computer Studies Standard Grade - Literacy ) Consumer Studies Dance Studies Design Dramatic Arts
Economics Electrical Technology Engineering Graphics & Design Geography History Hospitality Studies ( previously Home Economics ) Information Technology (
previously Computer Studies Higher Grade - Programming ) Life Sciences ( previously Biology ) Mechanical Technology Music Physical Science Religion Studies
Second Additional Language Third Additional Language Tourism Visual Arts Life Orientation ( edit ) Life Orientation ( colloquially abbreviated as `` LO '' ) has
been introduced into the senior high school phase as an examination subject . Life Orientation is a broad - learning subject that covers non-academic skills
needed in everyday life .
</item 2>

<item 3>
National Senior Certificate
Life Orientation is a broad - learning subject that covers non-academic skills needed in everyday life . Life Orientation is examined , marked and moderated
internally and comprises the following sections : World of Work , helping learners find guidance in their choice of career and prepare them for the working
world Health Education , this sections includes sex education , HIV / Aids talks , pregnancy , etc . Physical Education , involves physical exercise and
training , it is done on a separate basis but eventually forms part of the Life Orientation mark Religious Education , each South African school decides which
religion to adhere to ( as according to the South African schools act ) , learners may be exempted from these classes . In practice this means that learners may
be exempted from , example , the Bible Studies class , but are compelled to attend the Religious Studies class ( which deals with all religions as a study ) .
Citizenship and Responsibility , this section teaches learners about their rights and their duties as citizens .
</item 3>

<item 4>
National Senior Certificate
To study for a bachelor 's degree at a South African university requires that the applicant has at least an NSC endorsed by Umalusi , with a pass of 30 % in the
chosen university 's language of learning and teaching , as well as a level 4 or higher in the following list of designated , 19 - credit subjects : Accounting
Agricultural Sciences Business Studies Dramatic Arts Economics Engineering Graphics and Design Geography History Consumer Studies Information Technology
Languages ( one language of learning and teaching at a higher education institution and two other recognised language subjects ) Life Sciences Mathematics
Mathematical Literacy Music Physical Sciences Religion Studies Visual Arts See also ( edit ) High school # South Africa Matriculation in South Africa
Independent Examinations Board References ( edit ) Jump up ^ SOUTH AFRICAN NATIONAL ASSEMBLY : QUESTION 757 , DATE OF PUBLICATION OF INTERNAL QUESTION PAPER :
11 / 03 / 2011 ( INTERNAL QUESTION PAPER : 06 / 2011 ) Jump up ^ SOUTH AFRICAN NATIONAL ASSEMBLY : WRITTEN REPLY TO QUESTION 469 , DATE OF PUBLICATION IN
INTERNAL QUESTION PAPER : 26 - 03 - 2007 ( INTERNAL QUESTION PAPER NO.
</item 4>

<item 5>
National Senior Certificate
Matric results are usually out of 400 marks . Level system ( 1 to 7 ) ( edit ) Level 7 : 80 - 100 % ( Outstanding achievement ) Level 6 : 70 - 79 % (
Meritorious achievement ) Level 5 : 60 - 69 % ( Substantial achievement ) Level 4 : 50 - 59 % ( Moderate achievement ) Level 3 : 40 - 49 % ( Adequate
achievement ) Level 2 : 30 - 39 % ( Elementary achievement ) Level 1 : 0 - 29 % ( Not achieved - Fail ) The 5 % language compensation ( edit ) Language
compensation is described by several sources : `` To the final mark is added the language compensation , which is 5 % of the mark attained by the candidate for
all non-language subjects , for candidates whose mother tongue is not English or Afrikaans . The 5 % compensates learners for the disadvantage suffered by these
candidates being instructed in a language that is not their mother tongue . ''
</item 5>

<item 6>
National Senior Certificate
There are three pass levels , each with different pass requirements. First is a pass with Higher certificate , one must obtain 40 % in home language , 40 % in
two other subjects and must obtain 30 % in three other subjects . A learner who passed matric with higher certificate level , can not enroll for a University
Degree nor a Diploma at an institute of higher learning . The mean mark in any subject is usually about 55 . Only a small proportion of candidates score an ' A
' in any subject ( from as little as 2 % to a maximum of about 10 % in subjects taken by highly select groups . ) A further 8 -- 15 % are likely to gain a ' B '
and about 20 -- 25 % achieve a ' C ' grade . The National Senior Certificate is a group certificate and records an aggregate mark . The Department of Basic
Education has responsibility for general educational policy to be implemented by nine provincial education departments and private providers such as the
Independent Examinations Board ( IEB ) . There are nine provincial examination boards and three independent boards , of which the IEB is the biggest . The IEB
operates on a national level catering primarily for independent schools .
</item 6>

<item 7>
National Senior Certificate
Citizenship and Responsibility , this section teaches learners about their rights and their duties as citizens . Assessment ( edit ) There are three types of
subjects : General Subjects An end - of - year examination : 75 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Practical
Subjects An end - of - year examination : 50 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Practical component : 25 % of the
total mark Language Subjects An end - of - year examination : 50 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Oral
Examination : 25 % of the total mark Continuous Assessment ( CASS ) includes all the tests , examinations , tasks , activities , orals and projects done
throughout the year . Matric results are usually out of 400 marks .
</item 7>

<item 8>
National Senior Certificate
For example a learner who obtains a mark of zero ( 0 ) out of 300 will obtain 5 % of zero ( which is zero ) for language compensation ; a learner who obtains 10
out of 300 will receive 5 % of 10 , which is 0 , 5 marks , for language compensation ; a learner who obtains 100 out of 300 marks will obtain an additional 5
marks for language compensation . '' This kind of compensation has some impact on pass rates , but does provide significant impact at the upper end of the scale
, affecting those applying for admission to university . For example , a qualifying learner obtaining 95 % would receive 95 x 1.05 = 99.75 % ( which rounds to
100 % ) . A learner obtaining 40 % would receive 40 x 1.05 = 42.0 % . University entrance ( edit ) In addition to minimum grades required in each subject ,
universities either set their own entrance tests and / or use the National Benchmark Tests ( NBT ) .
</item 8>

<item 9>
National Senior Certificate
: 10 - 2007 ) Jump up ^ Politicsweb , 3 January 2011 , `` Umalusi explains matric mark adjustments '' , Dr Sizwe Mabizela , 03 January 2011 Jump up ^
http://nbt.ac.za/ About NBT Retrieved 21 March 2011 Jump up ^ http://wced.pgwc.gov.za/circulars/circulars08/e33_08.html External links ( edit ) Department of
Education official site IEB official site Retrieved from `` https://en.wikipedia.org/w/index.php?title=National_Senior_Certificate&oldid=837110357 '' Categories
: Education in South Africa Secondary school qualifications Hidden categories : Articles needing additional references from April 2010 All articles needing
additional references Talk Contents About Wikipedia Add links This page was last edited on 18 April 2018 , at 19 : 39 . About Wikipedia
</item 9>

<item 10>
National Senior Certificate
National Senior certificate - wikipedia National Senior certificate Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( April 2010 ) ( Learn how and
when to remove this template message ) The National Senior Certificate or NSC is a high school diploma and is the main school - leaving certificate in South
Africa . This certificate is commonly known as the matriculation ( matric ) certificate , as grade 12 is the matriculation grade . The NSC , previously known as
the Further Education and Training Certificate or FETC , replaced the Senior Certificate with effect from 2008 and was phased in starting with grade 10 in 2006
. Pupils study at least seven subjects , including two compulsory official South African languages , either Mathematics or Mathematical Literacy , Life
Orientation and three elective subjects . Subjects are all taken on the same level - there is no higher or standard grade as in the past . There are three pass
levels , each with different pass requirements.
</item 10>
=== END ITEM id=a5f8f3ed05c9a1d4 ===

=== ITEM id=24bc2a642fbcdad3 ===
Gold sentences and facts:
Sentence 0: Other applications are short - circuit - proof extra-low voltage transformers for toys or doorbell installations .
  Fact 0.0: Other applications are short-circuit-proof extra-low voltage transformers for toys.
  Fact 0.1: Other applications are short-circuit-proof extra-low voltage transformers for doorbell installations.

Retrieved items:
<item 1>
Transformer types
The epoxy protects the windings from dust and corrosive atmospheres . However , because the molds for casting the coils are only available in fixed sizes , the
design of the transformers is less flexible , which may make them more costly if customized features ( voltage , turns ratio , taps ) are required . Isolating
transformer ( edit ) An isolation transformer links two circuits magnetically , but provides no metallic conductive path between the circuits . An example
application would be in the power supply for medical equipment , when it is necessary to prevent any leakage from the AC power system into devices connected to
a patient . Special purpose isolation transformers may include shielding to prevent coupling of electromagnetic noise between circuits , or may have reinforced
insulation to withstand thousands of volts of potential difference between primary and secondary circuits . Instrument transformer ( edit ) Main article :
Instrument transformer Instrument transformers Instrument transformers are typically used to operate instruments from high voltage lines or high current
circuits , safely isolating measurement and control circuitry from the high voltages or currents . The primary winding of the transformer is connected to the
high voltage or high current circuit , and the meter or relay is connected to the secondary circuit .
</item 1>

<item 2>
Transformer types
Audio transformers are those specifically designed for use in audio circuits to carry audio signal . They can be used to block radio frequency interference or
the DC component of an audio signal , to split or combine audio signals , or to provide impedance matching between high impedance and low impedance circuits ,
such as between a high impedance tube ( valve ) amplifier output and a low impedance loudspeaker , or between a high impedance instrument output and the low
impedance input of a mixing console . Audio transformers that operate with loudspeaker voltages and current are larger than those that operate at microphone or
line level , which carry much less power . Bridge transformers connect 2 - wire and 4 - wire communication circuits . Being magnetic devices , audio
transformers are susceptible to external magnetic fields such as those generated by AC current - carrying conductors . `` Hum '' is a term commonly used to
describe unwanted signals originating from the `` mains '' power supply ( typically 50 or 60 Hz ) . Audio transformers used for low - level signals , such as
those from microphones , often include magnetic shielding to protect against extraneous magnetically coupled signals .
</item 2>

<item 3>
Transformer types
1 Uses 1.11 Resonant transformer 1.11. 1 Constant voltage transformer 1.12 Ferrite core 1.12.
</item 3>

<item 4>
Transformer types
Loudspeaker transformer ( edit ) Loudspeaker transformer in old radio In the same way that transformers create high voltage power transmission circuits that
minimize transmission losses , loudspeaker transformers can power many individual loudspeakers from a single audio circuit operated at higher than normal
loudspeaker voltages . This application is common in public address applications . Such circuits are commonly referred to as constant - voltage speaker systems
. Such systems are also known by the nominal voltage of the loudspeaker line , such as 25 - , 70 - and 100 - volt speaker systems ( the voltage corresponding to
the power rating of a speaker or amplifier ) . A transformer steps up the output of the system 's amplifer to the distribution voltage . At the distant
loudspeaker locations , a step - down transformer matches the speaker to the rated voltage of the line , so the speaker produces rated nominal output when the
line is at nominal voltage . Loudspeaker transformers commonly have multiple primary taps to adjust the volume at each speaker in steps .
</item 4>

<item 5>
Transformer types
Measurement of high voltages is possible by the potential transformers . Combined instrument transformer ( edit ) Combined instrument transformer A combined
instrument transformer encloses a current transformer and a voltage transformer in the same transformer . There are two main combined current and voltage
transformer designs : oil - paper insulated and SF insulated . One advantage of applying this solution is reduced substation footprint , due to reduced number
of transformers in a bay , supporting structures and connections as well as lower costs for civil works , transportation and installation . Pulse transformer (
edit ) Bothhand TS6121A pulse transformer A pulse transformer is a transformer that is optimised for transmitting rectangular electrical pulses ( that is ,
pulses with fast rise and fall times and a relatively constant amplitude ) . Small versions called signal types are used in digital logic and telecommunications
circuits , often for matching logic drivers to transmission lines . Medium - sized power versions are used in power - control circuits such as camera flash
controllers . Larger power versions are used in the electrical power distribution industry to interface low - voltage control circuitry to the high - voltage
gates of power semiconductors .
</item 5>

<item 6>
Transformer types
Applications : Intermediate frequency ( IF ) transformer in superheterodyne radio receiver Tank transformers in radio transmitters Tesla coil CCFL inverter
Oudin coil ( or Oudin resonator ; named after its inventor Paul Oudin ) D'Arsonval apparatus Ignition coil or induction coil used in the ignition system of a
petrol engine Electrical breakdown and insulation testing of high voltage equipment and cables . In the latter case , the transformer 's secondary is resonated
with the cable 's capacitance . Constant voltage transformer ( edit ) See also : Voltage regulator § Constant - voltage transformer By arranging particular
magnetic properties of a transformer core , and installing a ferro - resonant tank circuit ( a capacitor and an additional winding ) , a transformer can be
arranged to automatically keep the secondary winding voltage relatively constant for varying primary supply without additional circuitry or manual adjustment .
Ferro - resonant transformers run hotter than standard power transformers , because regulating action depends on core saturation , which reduces efficiency .
The output waveform is heavily distorted unless careful measures are taken to prevent this . Saturating transformers provide a simple rugged method to stabilize
an AC power supply .
</item 6>

<item 7>
Transformer types
It acts then both as a voltage transformer and as a magnetic ballast . Other applications are short - circuit - proof extra-low voltage transformers for toys or
doorbell installations . Resonant transformer ( edit ) Further information : Resonant inductive coupling Further information : Tesla coil § Resonant transformer
A resonant transformer is a transformer in which one or both windings has a capacitor across it and functions as a tuned circuit . Used at radio frequencies ,
resonant transformers can function as high Q factor bandpass filters . The transformer windings have either air or ferrite cores and the bandwidth can be
adjusted by varying the coupling ( mutual inductance ) . One common form is the IF ( intermediate frequency ) transformer , used in superheterodyne radio
receivers . They are also used in radio transmitters . Resonant transformers are also used in electronic ballasts for gas discharge lamps , and high voltage
power supplies . They are also used in some types of switching power supplies . Here the short - circuit inductance value is an important parameter that
determines the resonance frequency of the resonant transformer .
</item 7>

<item 8>
Transformer types
Audio transformers were originally designed to connect different telephone systems to one another while keeping their respective power supplies isolated , and
are still commonly used to interconnect professional audio systems or system components , to eliminate buzz and hum . Such transformers typically have a 1 : 1
ratio between the primary and the secondary . These can also be used for splitting signals , balancing unbalanced signals , or feeding a balanced signal to
unbalanced equipment . Transformers are also used in DI boxes to convert high - impedance instrument signals ( e.g. , bass guitar ) to low impedance signals to
enable them to connect to a microphone input on the mixing console . A particularly critical component is the output transformer of a valve amplifier . Valve
circuits for quality reproduction have long been produced with no other ( inter-stage ) audio transformers , but an output transformer is needed to couple the
relatively high impedance ( up to a few hundred ohms depending upon configuration ) of the output valve ( s ) to the low impedance of a loudspeaker . ( The
valves can deliver a low current at a high voltage ; the speakers require high current at low voltage . ) Most solid - state power amplifiers need no output
transformer at all .
</item 8>

<item 9>
Transformer types
1 Constant voltage transformer 1.12 Ferrite core 1.12. 1 Planar transformer 1.13 Oil cooled transformer 1.14 Cast resin transformer 1.15 Isolating transformer 2
Instrument transformer 2.1 Current transformer 2.2 Voltage transformer or potential transformer 2.3 Combined instrument transformer 3 Pulse transformer 4 RF
transformer 4.1 Air - core transformer 4.2 Ferrite - core transformer 4.3 Transmission - line transformer 4.4 Balun 5 IF transformer 6 Audio transformer 6.1
Loudspeaker transformer 6.2 Output transformer 6.3 Small signal transformer 6.4 Interstage and coupling transformers 7 Other types 7.1 Transactor 7.2 Hedgehog
7.3 Variometer and variocoupler 7.4 Rotary transformer 7.5 Variable differential transformer 7.6 Resolver and synchro 8 See also 9 References Power transformers
( edit ) Laminated core ( edit ) Laminated core transformer This is the most common type of transformer , widely used in electric power transmission and
appliances to convert mains voltage to low voltage to power electronic devices .
</item 9>

<item 10>
Transformer types
This is a common method used in portable current measuring instruments but permanent installations use more economical types of current transformer . Voltage
transformer or potential transformer ( edit ) Voltage transformers ( VT ) , also called potential transformers ( PT ) , are a parallel connected type of
instrument transformer , used for metering and protection in high - voltage circuits or phasor phase shift isolation . They are designed to present negligible
load to the supply being measured and to have an accurate voltage ratio to enable accurate metering . A potential transformer may have several secondary
windings on the same core as a primary winding , for use in different metering or protection circuits . The primary may be connected phase to ground or phase to
phase . The secondary is usually grounded on one terminal . There are three primary types of voltage transformers ( VT ) : electromagnetic , capacitor , and
optical . The electromagnetic voltage transformer is a wire - wound transformer . The capacitor voltage transformer uses a capacitance potential divider and is
used at higher voltages due to a lower cost than an electromagnetic VT . An optical voltage transformer exploits the electrical properties of optical materials
. Measurement of high voltages is possible by the potential transformers .
</item 10>
=== END ITEM id=24bc2a642fbcdad3 ===
