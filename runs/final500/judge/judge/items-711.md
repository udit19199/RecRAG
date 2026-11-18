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

=== ITEM id=675d09e3b2bcf6ec ===
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
1 Notes 2.3 Series 2 ( 2017 -- 18 ) 2.3. 1 Notes 2.4 Series 3 3 Production 3.1 History 3.2 Legal issues 3.3 Filming 3.4 Promotion 3.5 Sponsorship 3.6 Technical
Issues 4 Reception 5 Broadcast 6 See also 7 References 8 External links Format ( edit ) The Grand Tour is presented as a mix of pre-recorded television films
and live - audience segments presented in a large tent to an audience of about 300 members . The show and its segments are hosted by Jeremy Clarkson , Richard
Hammond , and James May . The show is presented from a pair of large tents on a Grand Tour around the world . The presenters sit around a trestle table , with
the studio audience seated in front of them . Films ( edit ) Each show is generally bookended by two television films , or two parts of the same film , produced
by one or more of the hosts .
</item 1>

<item 2>
The Grand Tour (TV series)
Studio audience segments for the first series were filmed in the travelling tent between 17 July and 11 December 2016 in various countries , while for the
second series the tent was stationed permanently in the Cotswolds . Series one was filmed over the course of mid - to late 2016 and premiered on 18 November
2016 . Series two was filmed over the same time period the following year ; however , filming was set back due to various illnesses and injuries suffered by the
presenters . Due to this , it premiered on 8 December 2017 . In the days following the launch , `` The Holy Trinity '' became Amazon Video 's most watched
premiere episode . Overall , the show has received positive reviews from critics . Contents ( hide ) 1 Format 1.1 Films 1.2 Test tracks 1.2. 1 Eboladrome 1.2. 2
Enstone 1.3 Celebrity Face Off 1.4 Conversation Street 1.5 Running gags 1.6 Awards 2 Episodes 2.1 Series overview 2.2 Series 1 ( 2016 -- 17 ) 2.2.
</item 2>

<item 3>
Graph query row:
episode: The Falls Guys
guest: Kevin Pietersen
relationship: RELATED_TO
connection: None
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
Rotten Tomatoes . Retrieved 17 July 2017 . Jump up ^ `` The Grand Tour season 2 episode 1 review : We 're in safe hands as Clarkson , Hammond and May return ''
. Digital Spy . Retrieved 8 December 2017 . Jump up ^ `` Television review : The Grand Tour on Amazon Prime '' . The Times . Retrieved 8 December 2017 . Jump up
^ `` The Grand Tour season 2 , episode 1 review : Has Jeremy Clarkson finally found his groove ? '' . The Daily Telegraph . Retrieved 8 December 2017 . Jump up
^ Clarkson , Jeremy ( 8 December 2017 ) . `` Jeremy Clarkson on Twitter '' . twitter.com . Retrieved 15 December 2017 . Jump up ^ Ling , Thomas ( 27 November
2017 ) . `` The Grand Tour 's first celebrity guests for series 2 have been revealed -- and there 's not a single woman '' . Retrieved 20 December 2017 . Jump
up ^ Long , Trevor .
</item 5>

<item 6>
The Grand Tour (TV series)
The Sun . Retrieved 23 December 2017 . ^ Jump up to : Daly , Emma ( 26 November 2016 ) . `` Jeremy Clarkson forced to ' kill off ' celebs on The Grand Tour
after BBC ' bans Top Gear rip - off star interviews ' '' . The Sun . Retrieved 26 November 2016 . Jump up ^ Lee , Kristen ( 16 November 2017 ) . `` ' Celebrity
Brain Crash ' Axed on the Grand Tour Season 2 '' . jalopnik.com . Retrieved 17 November 2017 . Jump up ^ McCreesh , Louise ( 27 November 2017 ) . `` The Grand
Tour season 2 confirms celebrity guests including The Hoff , Jack Bauer and Luke Evans '' . Digital Spy . Retrieved 28 November 2017 . Jump up ^ Hsieh , Tony (
20 November 2016 ) . `` The Grand Tour : 3 Mustangs from the Opening Sequence of the Premiere Episode '' . Grand Tour Nation . Retrieved 24 November 2016 . ^
Jump up to : `` Amazon Instant Video '' .
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
Graph query row:
episode: The Falls Guys
guest: Brian Wilson
relationship: RELATED_TO
connection: None
</item 8>

<item 9>
The Grand Tour (TV series)
For episode 2 , the list included 5.11 Tactical . A DHL Boeing 757 was featured in the opening sequence of episode 5 , the tent was located in Rotterdam , and
the DHL logo is featured on part of the crash barrier at the Eboladrome . Technical issues ( edit ) During filming for some episodes in the second series of the
show , the production team suffered major technical issues with the tent lighting , which threatened to derail filming . Nonetheless , all episodes were
released on the correct dates , with all delays having been averted . Reception ( edit ) As of November 2016 the show has received positive reviews from critics
, with The Guardian saying `` Jeremy Clarkson and co leave the BBC in their dust '' . Daily Express TV reporter , Neela Debnath commented that the first episode
`` resembled a Hollywood blockbuster '' and added that `` ( The Grand Tour is ) basically Top Gear on steroids '' . However , BBC Arts Editor , Will Gompertz
said of the opening that `` there is no irony .
</item 9>

<item 10>
The Grand Tour (TV series)
The Daily Telegraph , while not as positive , still approved of the episode , stating `` The writing is still rather ropey . Clarkson 's suggestion of a new
nickname for May -- `` Dingleberry Handpump '' -- failed to raise a titter even among the super-fans gathered for the London premiere '' but also said that ``
for each wobble , there are just as many moments when The Grand Tour manages the clever trick Top Gear could pull off at its best : raising a chuckle while
sneaking in a bit of serious journalism at the same time . '' and ultimately gave the episode 3 out of 5 stars . Jeremy Clarkson himself believed that they had
`` hit the ground running with series 2 of the Grand Tour '' . Radio Times has criticised the gender balance of guests on the `` Celebrity Face Off '' segment .
There was only 1 woman guest in the episode `` Oh Canada '' ; Paris Hilton . Hothouse Flowers playing for the opening sequence of `` The Holy Trinity '' episode
in the Lucerne Valley , California Broadcast ( edit ) Australian free - to - air network Seven Network started broadcasting the first series of The Grand Tour
in mid-October 2017 .
</item 10>
=== END ITEM id=675d09e3b2bcf6ec ===

=== ITEM id=7e26772c9c60110d ===
Gold sentences and facts:
Sentence 0: Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions , Groovy Solutions ( Series 3 -- 4 ) Josie Jump Buki Akib ( series 1 - 2 ) Kasia Haddad ( Series 3 -- 4 ) Fitness instructor Yellow house Josie Jump is My Name ( Jump a Little Higher ) ( Series 1 - 4 ) Cheer You Up ( Series 3 -- 4 ) Edie McCredie Juliet Cadzow Bus driver Blue garage / house When I Honk My Horn ( Series 1 - 4 ) Let Me Take You on a Journey ( Series 3 -- 4 ) PC Plum Andrew Agnew Policeman Black and white police station I 'm PC Plum ( Series 1 -- 4 ) Follow the Clue ( Series 3 -- 4 ) Spencer the Painter Rodd Christensen Painter / Musician Orange house Climbing Up My Musical Ladder ( Series 1 -- 4 ) If You Need a Little Rhythm ( Series 3 -- 4 ) Suzie Sweet Mary Riggans Shop and café workers Red shop I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Suzie 's Cooking ( Series 3 -- 4 ) Penny Pocket Kim Tserkezie I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Sort It ( Series 3 -- 4 )
  Fact 0.0: Miss Hoolie is played by Julie Wilson Nimmo.
  Fact 0.1: Miss Hoolie's job is nursery teacher.
  Fact 0.2: Miss Hoolie's house is the Green house (Nursery).
  Fact 0.3: Miss Hoolie's Nursery Song is Everybody everyone.
  Fact 0.4: Miss Hoolie's Day off song is What do you want to do today (Series 1-2).
  Fact 0.5: Miss Hoolie's Day off song is Strike up the band (Series 3-4).
  Fact 0.6: Archie the Inventor is played by Miles Jupp.
  Fact 0.7: Archie the Inventor's job is inventor.
  Fact 0.8: Archie the Inventor's house is the Pink castle.
  Fact 0.9: Archie the Inventor's song is I'm Archie the Inventor (Series 1-4).
  Fact 0.10: Archie the Inventor's song is Great Inventions, Groovy Solutions (Series 3-4).
  Fact 0.11: Josie Jump is played by Buki Akib in Series 1-2.
  Fact 0.12: Josie Jump is played by Kasia Haddad in Series 3-4.
  Fact 0.13: Josie Jump's job is fitness instructor.
  Fact 0.14: Josie Jump's house is the Yellow house.
  Fact 0.15: Josie Jump's song is Josie Jump is My Name (Jump a Little Higher) (Series 1-4).
  Fact 0.16: Josie Jump's song is Cheer You Up (Series 3-4).
  Fact 0.17: Edie McCredie is played by Juliet Cadzow.
  Fact 0.18: Edie McCredie's job is bus driver.
  Fact 0.19: Edie McCredie's house is the Blue garage / house.
  Fact 0.20: Edie McCredie's song is When I Honk My Horn (Series 1-4).
  Fact 0.21: Edie McCredie's song is Let Me Take You on a Journey (Series 3-4).
  Fact 0.22: PC Plum is played by Andrew Agnew.
  Fact 0.23: PC Plum's job is policeman.
  Fact 0.24: PC Plum's house is the Black and white police station.
  Fact 0.25: PC Plum's song is I'm PC Plum (Series 1-4).
  Fact 0.26: PC Plum's song is Follow the Clue (Series 3-4).
  Fact 0.27: Spencer the Painter is played by Rodd Christensen.
  Fact 0.28: Spencer the Painter's job is painter and musician.
  Fact 0.29: Spencer the Painter's house is the Orange house.
  Fact 0.30: Spencer the Painter's song is Climbing Up My Musical Ladder (Series 1-4).
  Fact 0.31: Spencer the Painter's song is If You Need a Little Rhythm (Series 3-4).
  Fact 0.32: Suzie Sweet is played by Mary Riggans.
  Fact 0.33: Suzie Sweet's job is shop and café worker.
  Fact 0.34: Suzie Sweet's house is the Red shop.
  Fact 0.35: Suzie Sweet's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.36: Suzie Sweet's song is Suzie's Cooking (Series 3-4).
  Fact 0.37: Penny Pocket is played by Kim Tserkezie.
  Fact 0.38: Penny Pocket's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.39: Penny Pocket's song is Sort It (Series 3-4).

Retrieved items:
<item 1>
Balamory
Each character also has one or more individual songs . Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery
Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp
Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions ,
</item 1>

<item 2>
Balamory
It was also shown on TLC . In Ireland , episodes were shown until recently on weekdays on RTÉ Two 's The Den and continue to be shown on The Den 's replacement
, RTÉjr . Tobermory , Mull , the setting for the fictional town of Balamory Edie McCredie 's bus ( LDV Convoy ) in the Glasgow Museum of Transport Contents (
hide ) 1 Characters 2 Episodes 2.1 Episode guide 3 Audience 4 Songs 5 Production 6 Retirement 7 Awards 8 References 9 External links Characters ( edit ) There
are eight central characters in Balamory . Each character wears a distinctive colour of clothing like pink or yellow , and lives in a house of the same colour
like Archie lives in a pink castle , with the exception of Penny Pocket ( who shares the red house / shop with Suzie Sweet , and wears blue ) and PC Plum ( who
lives in the white house but wears a police uniform ) . Each character also has one or more individual songs .
</item 2>

<item 3>
Balamory
Retrieved 13 January 2010 . External links ( edit ) http://www.bbc.co.uk/cbeebies/shows/balamory Press release at the launch of Balamory ) at bbc.co.uk
http://www.bbc.co.uk/pressoffice/pressreleases/stories/2002/08_august/30/balamory.shtml CBeebies - Balamory ) at bbc.co.uk Balamory on IMDb Balamory at TV.com
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Balamory&oldid=807129679 '' Categories : Balamory BBC children 's television programmes Fictional
populated places in Scotland 2002 British television programme debuts 2005 British television programme endings 2000s Scottish television series Musical
television series Preschool education television series Television shows set in Scotland Isle of Mull 2002 establishments in Scotland BBC Scotland television
programmes 2000s British children 's television series Hidden categories : Use dmy dates from March 2014 Articles needing additional references from December
2010 All articles needing additional references Talk Contents About Wikipedia Dansk Polski Edit links This page was last edited on 26 October 2017 , at 02 : 40
.
</item 3>

<item 4>
Balamory
Balamory - wikipedia Balamory Jump to : navigation , search This article needs additional citations for verification . Please help improve this article by
adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2010 ) ( Learn how and when to remove this template message
) Balamory Created by Brian Jameson Starring Julie Wilson Nimmo Miles Jupp Kasia Haddad Buki Akib Juliet Cadzow Andrew Agnew Rodd Christensen Mary Riggans Kim
Tserkezie Country of origin United Kingdom Original language ( s ) English No. of episodes 254 , including Christmas special . Production Producer ( s ) Brian
Jameson Running time 19 -- 21 minutes . 28 -- 30 minutes .
</item 4>

<item 5>
Balamory
This episode also featured Greg Hemphill ( Julie Wilson Nimmo 's husband in real life ) and John Altman , who plays Nick Cotton in the BBC soap opera EastEnders
. Celebrity chef Keith Floyd appeared in Suzie Sweet 's song `` Suzie 's Cookin ' '' . Occasionally , the shots of children entering the nursery are cut down
for repeats for time reasons . Retirement ( edit ) On 30 April 2005 , the BBC announced that it would not order another season of the series , citing they
wanted to experiment and create more new programme ideas . In that year , all the cast ( except for two of them ) performed a farewell touring show called ``
Balamory Live : Strike up the Band '' , a follow up to the `` Balamory Live : What 's The Story '' tour from 2004 . By the time the programme was retired , many
of the cast were also keen to move on . Miles Jupp , who played Archie , stated that his role caused some problems when he was performing stand up on the
Edinburgh Fringe , due to parents assuming his show was child friendly , and bringing their children to see it .
</item 5>

<item 6>
Balamory
Edie McCredie drives the nursery school bus . She has travelled all over the world , and has a variety of photographs and souvenirs to talk about . She is also
an expert driver and motor mechanic , and occasionally gets exasperated with other road users . Police officer PC Plum aspires to be a master detective , but
Balamory is a crime free village . He spends most of his time watching wildlife and using his detective skills , to figure out what the animals have been doing
. He is also particularly interested in rules , regulations , and safety . Spencer is officially responsible for painting the coloured houses of Balamory , but
also paints pictures and creates music . His painter 's ladder is musical , each rung sounding a different note . Spencer is also the only American to reside in
Balamory and he is black . Suzie Sweet and Penny Pocket together run the village shop and cafe . Suzie Sweet is an older , grandmotherly figure who loves to
tell customers about the items in the shop . Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics .
</item 6>

<item 7>
Balamory
She is advised to visit PC Plum , who advises her to feed it vegetables . She then goes to the shop run by Pocket and Sweet to buy rabbit food , and finally
brings the rabbit to Miss Hoolie 's nursery for the children to feed and pat it . Each episode of Balamory contains at least five songs : `` What 's the Story
in Balamory ? '' , the theme song Miss Hoolie 's opening song , either `` Everybody , Everyone '' if the nursery school is open , or one of the two `` day off
'' songs otherwise `` Which Coloured House Are We Going To ? '' , sung by the main character after presenting his or her problem Character - specific songs ,
sung by the characters visited by the main character Episode guide ( edit ) Series One Series Two Series Three Series Four Audience ( edit ) The programme was
aimed at pre schoolers . The programme 's creator had intended it as a `` soap opera for children '' . The show can also be seen in a humorous light . The
characters in Balamory are somewhat naive and light hearted , often appearing to be children trapped in adult bodies with adult roles .
</item 7>

<item 8>
Balamory
Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics . Episodes ( edit ) A typical episode follows this general plan : The
opening credits show the town of `` Balamory '' with its brightly coloured houses . Miss Hoolie opens the nursery school , greets the nursery children , and
talks about that day 's weather . One of the other characters ( who will be the main character of that episode ) tells Miss Hoolie about a problem . This
character then sings `` Which Coloured House Are We Going To ? '' , asking the young viewers to guess which of the Balamory characters will be most helpful in
solving the problem . The main character then proceeds to visit several other characters until the problem is solved . Miss Hoolie and the main character
provide a summary of the story at each point , and a final recap at the end of the episode . For example , in one episode Josie Jump has been asked to look
after a neighbour 's rabbit , but does not know what it needs . She is advised to visit PC Plum , who advises her to feed it vegetables .
</item 8>

<item 9>
Balamory
28 -- 30 minutes . Specials Production company ( s ) BBC Scotland Distributor BBC Release Original network CBeebies Picture format 16 : 9 Widescreen Audio
format Stereo Original release 10 February 2002 -- 14 April 2005 External links Website www.bbc.co.uk/cbeebies/shows/balamory Balamory is a Scottish live action
television series on British television ( BBC One , BBC Two and CBeebies ) for pre school children , about a fictional small island community called Balamory in
Scotland . It was produced between 10 February 2002 and 14 April 2005 by BBC Scotland , with 254 episodes made ( including a DVD exclusive Christmas episode ) .
The series was created by Brian Jameson . Balamory was filmed mostly in Tobermory on the Isle of Mull , with the exception of scenes at Archie 's castle filmed
at Fenton Tower in North Berwick , and other scenes , such as Camden and the nursery filmed in Glasgow . A select series of episodes was shown for two brief
seasons in the United States on the Discovery Kids channel . It was also shown on TLC .
</item 9>

<item 10>
Balamory
The storyline revolves around simple problems designed to challenge pre schoolers ' minds . Songs ( edit ) There are many songs in Balamory , mainly sung by the
actors . These songs are used consistently throughout the series , when corresponding characters or circumstances are introduced . In the run starting from
September 2004 , new songs were introduced . These songs are longer and take up more time in the episode , so the old versions were still often used . In
addition to the character specific songs mentioned in the `` Characters '' section , there are two general songs : `` What 's the Story in Balamory ? '' is the
show 's theme song , sung over the opening titles and played as an instrumental over the closing credits . `` Which Coloured House Are We Going To ? '' is sung
by the main character of each episode after introducing that episode 's problem . Production ( edit ) Very few episodes feature actors other than the main
characters and the nursery children . Terry Wogan made a guest appearance in one episode ( The Game Show ) as a television director .
</item 10>
=== END ITEM id=7e26772c9c60110d ===

=== ITEM id=737d5e59f4ede478 ===
Gold sentences and facts:
Sentence 0: The Sentencing Reform Act , part of the Comprehensive Crime Control Act of 1984 , was a U.S. federal statute intended to increase consistency in United States federal sentencing . It established the United States Sentencing Commission . It also abolished federal parole , except for persons convicted under federal law before November 1 , 1987 , persons convicted under District of Columbia law , `` transfer treaty '' inmates , persons who violated military law who are in federal civilian prisons , and persons who are defendants in state cases and who are under the U.S. Marshals Service Witness Protection Program .
  Fact 0.0: The Sentencing Reform Act was part of the Comprehensive Crime Control Act of 1984.
  Fact 0.1: The Sentencing Reform Act was a U.S. federal statute.
  Fact 0.2: The Sentencing Reform Act was intended to increase consistency in United States federal sentencing.
  Fact 0.3: The Sentencing Reform Act established the United States Sentencing Commission.
  Fact 0.4: The Sentencing Reform Act abolished federal parole.
  Fact 0.5: The abolition of federal parole did not apply to persons convicted under federal law before November 1, 1987.
  Fact 0.6: The abolition of federal parole did not apply to persons convicted under District of Columbia law.
  Fact 0.7: The abolition of federal parole did not apply to transfer treaty inmates.
  Fact 0.8: The abolition of federal parole did not apply to persons who violated military law who are in federal civilian prisons.
  Fact 0.9: The abolition of federal parole did not apply to persons who are defendants in state cases and who are under the U.S. Marshals Service Witness Protection Program.

Retrieved items:
<item 1>
The Senate Judiciary Subcommittee on Antitrust, Competition Policy and Consumer Rights oversees federal competition policy and enforcement, including the
Sherman, Clayton, and FTC Acts. In the 115th Congress, Mike Lee chaired it and Amy Klobuchar was ranking member. The 1984 Sentencing Reform Act established the
U.S. Sentencing Commission and largely ended federal parole, with exceptions. A separate article describes federal statutes: Congress has enacted over 20,000
since 1789; laws are published in the
</item 1>

<item 2>
Whistleblower Protection Act
Legislative history Introduced in the Senate as S. 20 by Carl Levin ( D -- MI ) on January 25 , 1989 Passed the Senate on March 16 , 1989 ( 97 - 0 , Roll call
vote 24 , via Senate.gov ) Passed the House on March 21 , 1989 ( Agreed voice vote ) Signed into law by President George H.W. Bush on April 10 , 1989 The
Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the
government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take )
retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file
complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority
; or a substantial and specific danger to public health or safety .
</item 2>

<item 3>
Sentencing Reform Act
eNotes . Retrieved 9 August 2011 . Jump up ^ `` FREQUENTLY ASKED QUESTIONS ABOUT THE LACK OF PAROLE FOR FEDERAL PRISONERS '' ( Archive ) . Families Against
Mandatory Minimums . p. 2 / 4 . Retrieved on December 13 , 2015 . Jump up ^ Stith , Kate ; Koh , Steve Y. ( 1993 ) . `` The Politics of Sentencing Reform : The
Legislative History of the Federal Sentencing Guidelines '' . Faculty Scholarship Series . Yale Law School . Retrieved 9 August 2011 . This United States
federal legislation article is a stub . You can help Wikipedia by expanding it .
</item 3>

<item 4>
California Highway Patrol
Of the incident , Ronald Reagan , who was governor of California at the time , said the following words : `` If anything worthwhile comes of this tragedy , it
should be the realization by every citizen that often the only thing that stands between them and losing everything they hold dear ... is the man wearing a
badge . '' An emotionally charged follow - up investigation followed the incident , but eventually led to a complete revision of procedures during high - risk
and felony stops . Firearms procedures have also changed fundamentally due to this incident , and physical methods of arrest have been improved . The police
baton and pepper spray have been added to the officer 's arsenal , with more in ‐ depth training in their use . The 25th anniversary of the Newhall Incident was
observed on April 6 , 1995 , at the present Newhall Area office , where a brick memorial pays tribute to Officers James Pence ( 6885 ) , Roger Gore ( 6547 ) ,
Walt Frago ( 6520 ) and George Alleyn ( 6290 ) .
</item 4>

<item 5>
Capital punishment in Australia
The last sentence of death in Western Australia ( indeed anywhere in Australia ) was passed in 1984 , but the female killer ( Brenda Hodge ) in question had her
sentence commuted to imprisonment for life , as was customary by this stage . Capital punishment was formally removed from the statutes of the state with the
passage of the Acts Amendment ( Abolition of Capital Punishment ) Act 1984 . South Australia ( edit ) The Adelaide Gaol was the site of forty - four hangings ,
from Joseph Stagg on 18 November 1840 to Glen Sabre Valance , murderer and rapist , on 24 November 1964 and three executions also occurred at Mount Gambier Gaol
. Two Ngarrindjeri men were controversially executed by hanging along the Coorong on 22 August 1840 , after a drumhead court - martial conducted by Police
Commissioner O'Halloran on the orders of Governor George Gawler . The men were found to be guilty of murdering the twenty - five survivors of the shipwreck
Maria a few months before .
</item 5>

<item 6>
On September 16, 1974, President Gerald Ford issued Proclamation 4313, creating a conditional amnesty program for Vietnam-era draft evaders who had fled abroad
and military deserters. Participants had to reaffirm loyalty to the United States and complete public service or additional military service, depending on their
prior service. A Clemency Board reviewed cases and recommended pardons and changes to discharge status; full pardons for draft evaders came under President
Carter. Ford initially retained Nixon’s Cabinet, but
</item 6>

<item 7>
Sentencing Reform Act
Sentencing Reform Act - Wikipedia Sentencing Reform Act Jump to : navigation , search The Sentencing Reform Act , part of the Comprehensive Crime Control Act of
1984 , was a U.S. federal statute intended to increase consistency in United States federal sentencing . It established the United States Sentencing Commission
. It also abolished federal parole , except for persons convicted under federal law before November 1 , 1987 , persons convicted under District of Columbia law
, `` transfer treaty '' inmates , persons who violated military law who are in federal civilian prisons , and persons who are defendants in state cases and who
are under the U.S. Marshals Service Witness Protection Program . The act was passed by large majorities in both houses of Congress . References ( edit ) Jump up
^ `` An Overview of the United States Sentencing Commission '' ( PDF ) . United States Sentencing Commission . United States Sentencing Commission . Retrieved 9
August 2011 . Jump up ^ Barry L. Johnson . `` Sentencing Reform Act ( 1984 ) '' . eNotes . Retrieved 9 August 2011 .
</item 7>

<item 8>
The references discuss the U.S. Bill of Rights, especially the Fourth through Eighth Amendments, including the exclusionary rule, criminal procedure, jury
trials, and cruel and unusual punishment. Sources include encyclopedias and a 2013 Congressional analysis of Supreme Court decisions.
</item 8>

<item 9>
National Flood Insurance Program
The CBRA enacted a set of maps depicting the John H. Chafee Coastal Barrier Resources System ( CBRS ) in which federal flood insurance is unavailable for new or
significantly improved structures . The National Flood Insurance Reform Act of 1994 codified the Community Rating System ( an incentive program that encourages
communities to exceed the minimal federal requirements for development within floodplains ) within the NFIP . The program was further amended by the Flood
Insurance Reform Act of 2004 , with the goal of reducing `` losses to properties for which repetitive flood insurance claim payments have been made . '' The
Biggert -- Waters Flood Insurance Reform Act of 2012 ( Biggert - Waters ) modified the NFIP . At the conclusion of 2011 , as Congress passed Biggert - Waters ,
the NFIP cumulative debt was over $17 billion . A core principle of Biggert - Waters was to change the NFIP premiums to match actuarial risk - based premiums
that better reflected the expected losses and real risk of flooding . These changes included removing discounts to many policies which were being sold below
actual actuarial risk targets and eliminating `` grandfathering '' of older rates .
</item 9>

<item 10>
Sentencing Reform Act
This United States federal legislation article is a stub . You can help Wikipedia by expanding it . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Sentencing_Reform_Act&oldid=813388767 '' Categories : United States federal criminal legislation Penal system in the
United States United States sentencing law 1984 in law 1984 in the United States United States federal legislation stubs Hidden categories : Use dmy dates from
August 2011 All stub articles Talk About Wikipedia Add links This page was last edited on 3 December 2017 , at 12 : 21 . About Wikipedia
</item 10>
=== END ITEM id=737d5e59f4ede478 ===

=== ITEM id=f1815b20e810cd7b ===
Gold sentences and facts:
Sentence 0: Douglas Irving Pederson ( born January 31 , 1968 ) is an American football coach who is currently the head coach of the Philadelphia Eagles of the National Football League ( NFL ) . He served as the offensive coordinator of the Kansas City Chiefs from 2013 -- 2015 . He spent most of his playing career as a member of the Green Bay Packers , serving as a backup quarterback to Brett Favre and holder on placekicks , and winning Super Bowl XXXI with the team over the New England Patriots . He was also a backup to Dan Marino as a member of the Miami Dolphins , and a starting quarterback for the Eagles and Cleveland Browns .
  Fact 0.0: Douglas Irving Pederson was born on January 31, 1968.
  Fact 0.1: Douglas Irving Pederson is an American football coach.
  Fact 0.2: Douglas Irving Pederson is currently the head coach of the Philadelphia Eagles.
  Fact 0.3: The Philadelphia Eagles are a team of the National Football League (NFL).
  Fact 0.4: Douglas Irving Pederson served as the offensive coordinator of the Kansas City Chiefs from 2013 to 2015.
  Fact 0.5: Douglas Irving Pederson spent most of his playing career as a member of the Green Bay Packers.
  Fact 0.6: Douglas Irving Pederson served as a backup quarterback to Brett Favre with the Green Bay Packers.
  Fact 0.7: Douglas Irving Pederson served as holder on placekicks with the Green Bay Packers.
  Fact 0.8: Douglas Irving Pederson won Super Bowl XXXI with the Green Bay Packers over the New England Patriots.
  Fact 0.9: Douglas Irving Pederson was also a backup to Dan Marino as a member of the Miami Dolphins.
  Fact 0.10: Douglas Irving Pederson was a starting quarterback for the Philadelphia Eagles.
  Fact 0.11: Douglas Irving Pederson was a starting quarterback for the Cleveland Browns.

Retrieved items:
<item 1>
Penn State Nittany Lions football
2013 , 2016 Current NFL players ( edit ) As of April 29 , 2017 : Player Team Position Adrian Amos Chicago Bears Cornerback Tarow Barney New York Jets Defensive
tackle Deion Barnes Linebacker Brad Bars Defensive end NaVorro Bowman Oakland Raiders Linebacker Glenn Carson Linebacker Kyle Carter Minnesota Vikings Tight end
Jack Crawford Atlanta Falcons Defensive end Sam Ficken Los Angeles Rams Placekicker Garry Gilliam Seattle Seahawks Offensive tackle Chris Godwin Tampa Bay
Buccaneers Wide receiver Robbie Gould San Francisco 49ers Placekicker Christian Hackenberg New York Jets Quarterback Tamba Hali Kansas City Chiefs Linebacker
Chris Hogan New England Patriots Wide receiver Jordan Hill Detroit Lions Defensive tackle Mike Hull Miami Dolphins Linebacker Gerald Hodges New Orleans Saints
Linebacker Jesse James Pittsburgh Steelers Tight end Austin Johnson Tennessee Titans Defensive tackle DaQuan Jones Tennessee Titans Defensive end Sean Lee
Dallas Cowboys Linebacker Jordan Lucas Miami Dolphins Safety Michael Mauti New Orleans Saints Linebacker Matt McGloin Quarterback Derek Moye Wide receiver Carl
Nassib Cleveland Browns Defensive end Jared Odrick Defensive end Rich Ohrnberger Guard Paul Posluszny Jacksonville Jaguars Linebacker Andrew Quarless Tight end
Allen Robinson Jacksonville Jaguars Wide receiver Travis Ross Kansas City Chiefs Tight end Evan Royster Running back A.
</item 1>

<item 2>
Doug Pederson
91 Fletcher Cox 93 Timmy Jernigan 94 Beau Allen 95 Mychal Kendricks 96 Derek Barnett 97 Destiny Vaeao 98 Elijah Qualls Randall Goforth Aziz Shittu Dom Williams
Head coach : Doug Pederson Coaches : Eugene Chung Phillip Daniels John DeFilippo Dave Fipp Ken Flajole Keith Gray Mike Groh Matthew Harper Tim Hauck Josh Hingst
Shaun Huls Trent Miles Ryan Paganetti T.J. Paganetti Justin Peelle Spencer Phillips Frank Reich Jim Schwartz Duce Staley Jeff Stoutland Press Taylor Cory Undlin
Dino Vasso Chris Wilson Philadelphia Eagles starting quarterbacks Red Kirkman ( 1933 -- 1935 ) Jim Leonard ( 1934 , 1936 ) Stumpy Thomason ( 1935 ) Irv Kupcinet
( 1935 ) Glenn Frey ( 1936 -- 1937 ) Jess Dow ( 1938 -- 1939 ) Davey O'Brien ( 1939 -- 1940 ) Len Barnum ( 1941 ) Tommy Thompson ( 1941 -- 1942 ,
</item 2>

<item 3>
Doug Pederson
3 Philadelphia Eagles 1.2. 4 Cleveland Browns 1.2. 5 Green Bay Packers ( second stint ) 2 Coaching career 2.1 High school 2.2 NFL assistant coaching positions
2.3 NFL head coach 2.4 Head coaching record 2.5 Coaching tree 3 Personal life 4 References 5 External links Playing career ( edit ) Early years ( edit )
Pederson was born in Bellingham , Washington , in 1968 . He attended Ferndale High School in nearby Ferndale , Washington , and was an All - State selection in
football , basketball , and baseball . After high school he graduated from Northeast Louisiana University , where he was quarterback from 1987 through 1990 . He
still holds multiple passing records at the school . Professional ( edit ) Miami Dolphins ( edit ) Pederson originally signed as a rookie free agent by the
Miami Dolphins on May 1 , 1991 , out of Northeast Louisiana University ( now University of Louisiana at Monroe ) in Monroe , Louisiana . He was waived on August
17 , 1991 , before the start of the regular season .
</item 3>

<item 4>
Doug Pederson
Bills ) Adam Gase ( Miami Dolphins ) Bill Belichick ( New England Patriots ) Todd Bowles ( New York Jets ) John Harbaugh ( Baltimore Ravens ) Marvin Lewis (
Cincinnati Bengals ) Hue Jackson ( Cleveland Browns ) Mike Tomlin ( Pittsburgh Steelers ) Bill O'Brien ( Houston Texans ) Frank Reich ( Indianapolis Colts )
Doug Marrone ( Jacksonville Jaguars ) Mike Vrabel ( Tennessee Titans ) Vance Joseph ( Denver Broncos ) Andy Reid ( Kansas City Chiefs ) Anthony Lynn ( Los
Angeles Chargers ) Jon Gruden ( Oakland Raiders ) National Football Conference NFC East NFC North NFC South NFC West Jason Garrett ( Dallas Cowboys ) Pat
Shurmur ( New York Giants ) Doug Pederson ( Philadelphia Eagles ) Jay Gruden ( Washington Redskins ) Matt Nagy ( Chicago Bears ) Matt Patricia ( Detroit Lions )
Mike McCarthy ( Green Bay Packers ) Mike Zimmer ( Minnesota Vikings ) Dan Quinn ( Atlanta Falcons ) Ron Rivera ( Carolina Panthers ) Sean Payton ( New Orleans
Saints ) Dirk Koetter ( Tampa Bay Buccaneers ) Steve Wilks ( Arizona Cardinals ) Sean McVay ( Los Angeles Rams ) Kyle Shanahan ( San Francisco 49ers ) Pete
Carroll ( Seattle Seahawks ) Links to related articles Louisiana -- Monroe Warhawks starting quarterbacks Lloyd Ray ( 1951 ) Mark
</item 4>

<item 5>
Doug Pederson
1965 ) Frank Ryan ( 1962 -- 1968 ) Gary Lane ( 1967 ) Bill Nelsen ( 1968 -- 1972 ) Mike Phipps ( 1970 -- 1976 ) Don Gault ( 1970 ) Brian Sipe ( 1974 -- 1983 )
Will Cureton ( 1975 ) Dave Mays ( 1977 ) Terry Luck ( 1977 ) Paul McDonald ( 1982 -- 1984 ) Gary Danielson ( 1985 -- 1988 ) Bernie Kosar ( 1985 -- 1993 ) Jeff
Christensen ( 1987 ) Mike Pagel ( 1988 , 1990 ) Don Strock ( 1988 ) Mike Tomczak ( 1992 ) Todd Philcox ( 1992 -- 1993 ) Vinny Testaverde ( 1993 -- 1995 ) Mark
Rypien ( 1994 ) Eric Zeier ( 1995 ) Ty Detmer ( 1999 ) Tim Couch ( 1999 -- 2003 ) Doug Pederson ( 2000 ) Spergon Wynn ( 2000 ) Kelly Holcomb ( 2002 --
</item 5>

<item 6>
Doug Pederson
85 Terry Mickens 86 Antonio Freeman 87 Robert Brooks 88 Keith Jackson 89 Mark Chmura 90 Darius Holland 91 Shannon Clavelle 92 Reggie White 93 Gilbert Brown 94
Bob Kuberski 95 Keith McKenzie 96 Sean Jones 98 Gabe Wilkins Head coach : Mike Holmgren Coaches : Larry Brooks Nolan Cromwell Gil Haskell Johnny Holland Sherman
Lewis Jim Lind Tom Lovat Marty Mornhinweg Andy Reid Gary Reynolds Fritz Shurmur Harry Sydney Bob Valesente Philadelphia Eagles Super Bowl LII champions 4 Jake
Elliott 6 Caleb Sturgis 7 Nate Sudfeld 8 Donnie Jones 9 Nick Foles ( MVP ) 10 Mack Hollins 11 Carson Wentz 13 Nelson Agholor 14 Marcus Johnson 16 Bryce Treggs
17 Alshon Jeffery 18 Shelton Gibson 19 Rashard Davis 21 Patrick Robinson 22 Sidney Jones 23 Rodney McLeod 24 Corey Graham 26 Jaylen Watkins 27 Malcolm Jenkins
28 Wendell Smallwood 29 LeGarrette Blount 30 Corey Clement 31 Jalen Mills 32 Rasul Douglas 33 De'Vante Bausby 34 Donnel
</item 6>

<item 7>
Doug Pederson
Doug Pederson - wikipedia Doug Pederson Jump to : navigation , search Not to be confused with Doug Peterson . Doug Pederson Pederson in 2016 Philadelphia Eagles
Position : Head coach Born : ( 1968 - 01 - 31 ) January 31 , 1968 ( age 50 ) Bellingham , Washington Height : 6 ft 3 in ( 1.91 m ) Weight : 221 lb ( 100 kg )
Career information High school : Ferndale ( WA ) College : Northeast Louisiana Undrafted : 1991 Career history As player : Miami Dolphins ( 1991 ) * New York /
New Jersey Knights ( 1992 ) Miami Dolphins ( 1992 -- 1994 ) Carolina Panthers ( 1995 ) * Rhein Fire ( 1995 ) Miami Dolphins ( 1995 ) Green Bay Packers ( 1995 --
1998 ) Philadelphia Eagles ( 1999 ) Cleveland Browns ( 2000 ) Green Bay Packers ( 2001 -- 2004 ) * Offseason and / or practice squad member only As coach :
Calvary Baptist Academy ( 2005 -- 2008 ) Head coach Philadelphia Eagles ( 2009 -- 2010 ) Offensive quality
</item 7>

<item 8>
Doug Pederson
Bustle Chris Scelfo Green Bay Packers Super Bowl XXXI champions 4 Brett Favre 7 Kyle Wachholtz 9 Jim McMahon 13 Chris Jacke 17 Craig Hentrich 18 Doug Pederson
21 Craig Newsome 25 Dorsey Levens 27 Calvin Jones 28 Roderick Mullen 30 William Henderson 32 Travis Jervey 33 Doug Evans 34 Edgar Bennett 36 LeRoy Butler 37
Tyrone Williams 39 Mike Prior 40 Chris Hayes 41 Eugene Robinson 46 Michael Robinson 51 Brian Williams 52 Frank Winters 53 George Koonce 54 Ron Cox 55 Bernardo
Harris 56 Lamont Hollinquest 59 Wayne Simmons 62 Marco Rivera 63 Adam Timmerman 64 Bruce Wilkerson 65 Lindsay Knapp 67 Jeff Dellenbach 68 Gary Brown 71 Santana
Dotson 72 Earl Dotson 73 Aaron Taylor 75 Ken Ruettgers 77 John Michels 80 Derrick Mayes 81 Desmond Howard ( MVP ) 82 Don Beebe 83 Jeff Thomason 84 Andre Rison
85 Terry Mickens 86 Antonio Freeman 87 Robert Brooks 88 Keith Jackson 89 Mark
</item 8>

<item 9>
Doug Pederson
-- 2008 ) Head coach Philadelphia Eagles ( 2009 -- 2010 ) Offensive quality control coordinator Philadelphia Eagles ( 2011 -- 2012 ) Quarterbacks coach Kansas
City Chiefs ( 2013 -- 2015 ) Offensive coordinator Philadelphia Eagles ( 2016 -- present ) Head coach Career highlights and awards As player Super Bowl champion
( XXXI ) As coach Super Bowl champion ( LII ) Career NFL statistics Pass attempts : 522 Pass completions : 286 Percentage : 54.7 TD -- INT : 12 -- 19 Passing
yards : 2,762 Passer rating : 62.3 Player stats at NFL.com Head coaching record Regular season : 20 -- 12 (. 625 ) Postseason : 3 -- 0 ( 1.000 ) Career : 23 --
12 (. 657 ) Coaching stats at PFR Douglas Irving Pederson ( born January 31 , 1968 ) is an American football coach who is currently the head coach of the
Philadelphia Eagles of the National Football League ( NFL ) . He served as the offensive coordinator of the Kansas City Chiefs from 2013 -- 2015 .
</item 9>

<item 10>
Doug Pederson
Stan Humphries ( 1986 -- 1987 ) Doug Pederson ( 1987 -- 1990 ) Wendal Lowrey ( 1991 -- 1992 ) Robert Cobb ( 1993 ) Raymond Philyaw ( 1993 -- 1996 ) Daniel
Jeremiah ( 1997 ) Andre Vige ( 1998 ) Andy Chance ( 1999 -- 2001 ) Steven Jyles ( 2002 -- 2005 ) Kinsmon Lancaster ( 2006 -- 2008 ) Trey Revell ( 2007 -- 2009 )
Cody Wells ( 2009 , 2012 ) Kolton Browning ( 2010 -- 2013 ) Brayle Brown ( 2013 ) Pete Thomas ( 2014 ) Garrett Smith ( 2015 -- 2017 ) Earnest Carrington ( 2015
) Will Collins ( 2016 ) Caleb Evans ( 2016 -- 2017 ) 1987 Northeast Louisiana Indians football -- NCAA Division I - AA national champions Teddy Garcia Jackie
Harris Stan Humphries Doug Pederson Head coach : Pat Collins Assistant coaches : Rickey Bustle Chris Scelfo Green Bay Packers Super Bowl XXXI champions 4 Brett
Favre
</item 10>
=== END ITEM id=f1815b20e810cd7b ===

=== ITEM id=3485b3199e304d12 ===
Gold sentences and facts:
Sentence 0: Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 - 08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco 49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 , 1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All - American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9 Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of Fame
  Fact 0.0: Deion Sanders is pictured in 2011.
  Fact 0.1: Deion Sanders' jersey numbers are No. 21 and No. 37.
  Fact 0.2: Deion Sanders' position is cornerback.
  Fact 0.3: Deion Sanders was born on August 9, 1967.
  Fact 0.4: Deion Sanders is listed as age 50.
  Fact 0.5: Deion Sanders was born in Fort Myers, Florida.
  Fact 0.6: Deion Sanders is 6 ft 1 in (1.85 m) tall.
  Fact 0.7: Deion Sanders weighs 198 lb (90 kg).
  Fact 0.8: Deion Sanders' high school is North Fort Myers (FL).
  Fact 0.9: Deion Sanders' college is Florida State.
  Fact 0.10: Deion Sanders was picked in round 1, pick 5 of the 1989 NFL Draft.
  Fact 0.11: Deion Sanders played for the Atlanta Falcons from 1989 to 1993.
  Fact 0.12: Deion Sanders played for the San Francisco 49ers in 1994.
  Fact 0.13: Deion Sanders played for the Dallas Cowboys from 1995 to 1999.
  Fact 0.14: Deion Sanders played for the Washington Redskins in 2000.
  Fact 0.15: Deion Sanders played for the Baltimore Ravens from 2004 to 2005.
  Fact 0.16: Deion Sanders is a 2-time Super Bowl champion, for Super Bowls XXIX and XXX.
  Fact 0.17: Deion Sanders is an 8-time Pro Bowl selection (1991-1994, 1996-1999).
  Fact 0.18: Deion Sanders is an 8-time First-team All-Pro (1991-1994, 1996-1999).
  Fact 0.19: Deion Sanders was NFL Defensive Player of the Year in 1994.
  Fact 0.20: Deion Sanders is named to the NFL 1990s All-Decade Team.
  Fact 0.21: Deion Sanders has an Atlanta Falcons Ring of Honor entry.
  Fact 0.22: Deion Sanders won the Jim Thorpe Award in 1988.
  Fact 0.23: Deion Sanders is a 2-time Unanimous All-American (1987, 1988).
  Fact 0.24: No. 2 is retired by the Florida State Seminoles.
  Fact 0.25: Deion Sanders' career NFL interceptions total 53.
  Fact 0.26: Deion Sanders' career NFL defensive touchdowns total 10.
  Fact 0.27: Deion Sanders' career NFL return touchdowns total 9.
  Fact 0.28: Deion Sanders' career NFL receptions total 60.
  Fact 0.29: Deion Sanders' career NFL receiving yards total 784.
  Fact 0.30: Deion Sanders' career NFL receiving touchdowns total 3.
  Fact 0.31: The page links to player stats at NFL.com.
  Fact 0.32: The page links to player stats at PFR.
  Fact 0.33: The page links to the Pro Football Hall of Fame.
  Fact 0.34: The page links to the College Football Hall of Fame.

Retrieved items:
<item 1>
Deion Sanders
NFL . National Football League . Retrieved 28 January 2017 . Jump up ^ `` Deion Sanders HOF Speech '' . Video Entry . Retrieved October 19 , 2011 . Jump up ^
Kallestad , Brent . `` Deion Changes the Rules Again : Pro football : This time , however , it 's for better education of athletes with new standards in Florida
. '' . Los Angeles Times . Retrieved 29 January 2017 . Jump up ^ Trex , Ethan . `` Two Rules Named After Deion Sanders '' . mental_floss . Mental Floss , Inc .
Retrieved 29 January 2017 . Jump up ^ `` NFL Videos : 2011 HOF : Deion Sanders '' . Nfl.com . August 6 , 2011 . Retrieved August 16 , 2012 . Jump up ^ Frohman ,
Jimmy . `` 11 Most Awesome NFL Player Nicknames '' . Phactual . Phactual , LLC . Retrieved 24 January 2017 . Jump up ^ `` CES ProNews Flashes ! '' .
</item 1>

<item 2>
Deion Sanders
Deion Sanders - wikipedia Deion Sanders Jump to : navigation , search Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 -
08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High
school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco
49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl
champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 ,
</item 2>

<item 3>
Deion Sanders
In 2014 , he debuted his new show , the Deion Family Playbook . Contents ( hide ) 1 Early life 1.1 High school 1.2 College career 2 Professional baseball career
3 Professional football career 3.1 Draft and Atlanta Falcons 3.2 San Francisco 49ers 3.3 Dallas Cowboys 3.4 Washington Redskins 3.5 Baltimore Ravens 4 Legacy 5
Media appearances and pop culture fame 6 Other business and entertainment ventures 7 Discography 8 Personal life 9 See also 10 References 11 External links
Early life ( edit ) High school ( edit ) Sanders was born in Fort Myers , Florida . He attended North Fort Myers High School in North Fort Myers , and was a
letterman and All - State honoree in football , basketball and baseball . In 1985 , Sanders was named to the Florida High School Association All - Century Team
which selected the top 33 players in the 100 - year history of high school football in the state . The Kansas City Royals selected Sanders out of North Fort
Myers High School in the sixth round of the 1985 Major League Baseball draft . He did not sign with the Royals .
</item 3>

<item 4>
Deion Sanders
He played in the National Football League ( NFL ) for the Atlanta Falcons , the San Francisco 49ers , the Dallas Cowboys , the Washington Redskins and the
Baltimore Ravens , winning the Super Bowl with both the 49ers and the Cowboys . An outfielder in baseball , he played professionally for the New York Yankees ,
the Atlanta Braves , the Cincinnati Reds and the San Francisco Giants , and participated in the 1992 World Series with the Braves . He attended Florida State
University , where he was recognized as a two - time All - American in football , and also played baseball and ran track . Sanders founded the Prime Prep
Academy charter school in 2012 and has coached at the school . The school was closed January 30 , 2015 , due to financial insolvency . In 2014 , he debuted his
new show , the Deion Family Playbook .
</item 4>

<item 5>
Deion Sanders
After getting into a scuffle with his former Falcon teammate Andre Rison , Sanders intercepted a pass from quarterback Jeff George and returned it 93 yards
while mockingly staring down the entire Falcons sideline before high - stepping into the end zone . Sanders was later voted the 1994 NFL Defensive Player of the
Year and recorded an interception in the 49ers 49 -- 26 victory over the San Diego Chargers in Super Bowl XXIX , earning him his first championship ring .
Dallas Cowboys ( edit ) Following his successful season with the 49ers , Sanders , along with his agent Eugene Parker , courted numerous teams in need of a
cornerback . The several teams in the `` Deion Sweepstakes '' , as it was called by the media , were the Philadelphia Eagles , Oakland Raiders , Miami Dolphins
, New Orleans Saints , San Francisco 49ers and the Dallas Cowboys , who had lost their starting cornerback Kevin Smith to injury for the rest of the season .
</item 5>

<item 6>
Deion Sanders
1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All -
American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9
Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of
Fame Deion Luywnn Sanders Sr. ( / ˈdiːɒn / ; born August 9 , 1967 ) , nicknamed Primetime , is a former American football and baseball player who works as an
analyst for CBS Sports and the NFL Network . He was inducted into the Pro Football Hall of Fame in 2011 . Sanders played football primarily at cornerback , but
also as a kick returner , punt returner and occasionally wide receiver .
</item 6>

<item 7>
Deion Sanders
In 1995 , he signed with the Dallas Cowboys for a minimum yearly base salary and a nearly $13 million signing bonus in an attempt to circumvent the NFL 's
salary cap . This caused the NFL to institute its own ' Deion Sanders rule ' whereby a prorated portion of a player 's signing bonus counted against the salary
cap . Media appearances and pop culture Fame ( edit ) See also : Leon Sandcastle Sanders became known for sporting a `` do - rag '' or bandana and for his high
- stepping into the end zone followed by his touchdown dance celebrations . At the end of his Hall of Fame speech , he put a bandana on his bust . His `` Prime
Time '' nickname was given to him by a friend and high - school teammate , Florida Gators defensive back Richard Fain . The two played pickup basketball games
together during the prime time television hour , and Sanders ' athletic display during those games earned him the nickname . His other nickname is `` Neon Deion
'' . Following his first Super Bowl victory with the San Francisco 49ers , Sanders hosted Saturday Night Live , broadcast on February 18 , 1995 .
</item 7>

<item 8>
Deion Sanders
Sanders amassed 7,838 all - purpose yards and scored 22 touchdowns : nine interception returns , six punt returns , three kickoff returns , three receiving ,
and one fumble recovery . His 19 defensive and return touchdowns was an NFL record ( now held by Devin Hester ) . In the postseason , Sanders added 5 more
interceptions , as well as 3 receptions for 95 yards , 4 carries for 39 yards , and two touchdowns ( one rushing and one receiving ) . He was selected to eight
Pro Bowls and won the NFL Defensive Player of the Year Award in 1994 . College Football News named Sanders # 8 in its list of 100 Greatest College Football
Players of All - Time . The Sporting News named Sanders # 37 in their Top 100 Football Players of the Century released in 1999 . ESPN named Sanders # 74 in its
list of the 100 Great Athletes of the Century released in 1999 . NFL.com named Sanders # 34 on NFL 's Top 100 list released in late 2010 On November 11 , 2010 ,
Sanders was inducted into the Atlanta Falcons ' Ring of Honor .
</item 8>

<item 9>
Deion Sanders
Sanders also occasionally lined up with his team 's offense . During the 1996 season , Sanders skipped the baseball season , concentrating on football , and
attended the first NFL training camp of his career to better familiarize himself with the nuances of the wide receiver position . He became only the second two
- way starter ( after the Cardinals ' Roy Green ) in the NFL since Chuck Bednarik for the first half of the season due . Sanders is the only man to play in both
a Super Bowl and a World Series , to hit an MLB home run and score an NFL touchdown in the same week , and to have both a reception and an interception in the
Super Bowl . He is one of two players to score an NFL touchdown six different ways . During his career , Sanders intercepted 53 passes for 1,331 yards ( a 25.1
yards per return average ) , recovered four fumbles for 15 yards , returned 155 kickoffs for 3,523 yards , gained 2,199 yards on 212 punt returns , and caught
60 passes for 784 yards .
</item 9>

<item 10>
Deion Sanders
On February 5 , 2011 , Sanders was announced as a Pro Football Hall of Fame inductee in his first year of eligibility . On May 17 , 2011 , Sanders was announced
as a College Football Hall of Fame inductee . On August 6 , 2011 , Sanders was inducted into the Pro Football Hall of Fame . Sanders also had a rushing TD in
the playoffs ( against the Philadelphia Eagles in January 1996 ) . This makes him ( including post season ) one of only two players in NFL history ( Bill Dudley
being the other ) to score a touchdown six different ways ( interception return , punt return , kickoff return , receiving , rushing , and a fumble recovery ) .
On February 6 , 2011 , at Super Bowl XLV in Arlington , Texas , Sanders performed the pre-game coin toss . Sanders did not attend classes or take final exams
during the fall semester of his senior year at Florida State , yet played in the Sugar Bowl against Auburn . This caused the Florida State Legislature to create
the ' Deion Sanders rule ' , whereby a football athlete at any state university could not play in a bowl game if he failed to successfully complete the previous
semester .
</item 10>
=== END ITEM id=3485b3199e304d12 ===

=== ITEM id=d7ea628d02a88626 ===
Gold sentences and facts:
Sentence 0: No . Title Original air date `` The Dark Dream Begins '' `` Kuroki Yume no Hajimari '' ( 黒き 夢 の はじまり ) August 28 , 2016 Elizabeth Liones goes to the Boar Hat , where all her friends are still asleep following the festival . After Elizabeth is greeted by Hawk outside , Meliodas introduces her to a meat pie made with vegetables that looks like Hawk , but it turns out to be unfit for human consumption . Meliodas has the idea of using pork as the main ingredient , which causes Hawk to make haste out of the bar . Meliodas comes up with a contest for his friends , in which he will obey any order for a day from whoever catches Hawk first . However , Hawk manages to outsmart them due to his small size . Hawk returns to the Boar Hat and eats food scraps made by Ban , who is declared the winner as a result . At night , Meliodas takes Elizabeth back to the palace while also assuring that his purpose in life is to protect her . Meliodas then goes to the catacombs to see the Horn of Cernunnos . When the goddess vessel taunts him for his reputation as a demon , Meliodas destroys the Horn of Cernunnos in retaliation . `` Our Fighting Festival '' `` Futari no Ken'ka Matsuri '' ( 二 人 の 喧嘩 祭り ) September 4 , 2016 Ban walks up to a mountaintop to finally settle matters with Meliodas , but the former is surprised to see Elizabeth and Hawk as well as the other members of the Seven Deadly Sins and the Holy Knights as part of an audience . Merlin casts a gigantic cubic barrier around Meliodas and Ban in order for the spectators to safely watch the fight . Meanwhile , King and Hauser have their own fight to win the attention of Diane . Ban initiates a brutal attack on Meliodas that somehow weakens most of the spectators . The fight ends in a stalemate after the two of them exchange rapid punches that cause the barrier to break . They remain friends and head back down to the Boar Hat for a drink . `` In Pursuit of First Love '' `` Hatsukoi o Oikakete '' ( 初恋 を 追いかけ て ) September 11 , 2016 King still hesitates to confess his feelings for Diane . Meanwhile , Diane leaves the Boar Hat and heads to a construction site in town , recalling the devastating collateral damage she caused due to her immense size . The construction workers there apologize to her and thank her for saving the kingdom . When King comes to see Diane , the two decide to play a game of tag across town . Outside a church , a wedding ceremony is interrupted by a demon , in which Diane and King combine their skills in defeating the demon . King asks Diane if she regained her memories , and she answers him with a kiss on his cheek . The moment is ruined when a brick suddenly falls on his head , and he later wakes up next to her , having forgotten what they did during the day . `` The Shape of Love '' `` Ai no Katachi '' ( 愛 の かたち ) September 18 , 2016 Merlin agrees to be a waitress at the Boar Hat , but first she wants Meliodas and Gowther to follow Gilthunder to find out if he is being stalked by someone . Meanwhile , Gilthunder , Hauser and Griamore are each hit with falling debris in the streets , unaware of the situation . Gowther is distracted by a boy named Pelliot , who asks him to retrieve a fat cat off a roof . Since Gowther reminds Pelliot of his mother , who recently passed away , Gowther transforms into Pelliot 's mother for a moment and disappears when Pelliot 's father finds Pelliot . Meliodas finds Gilthunder , Hauser and Griamore at a restaurant , where he purposely spills coffee to expose Vivian , revealed as the stalker . Merlin punishes Vivian by planting a cursed ring on her finger , which will bring pain to her each time the curse is activated , and Merlin explains that the Ten Commandments are the highest class in the demon race . At the Boar Hat , Merlin quits her job as a waitress , while Hawk and Gowther fill in for the position , much to the guys ' dismay .
  Fact 0.0: The episode 'The Dark Dream Begins' (Japanese title 'Kuroki Yume no Hajimari', 黒き 夢 の はじまり) first aired on August 28, 2016.
  Fact 0.1: Elizabeth Liones goes to the Boar Hat, where all her friends are still asleep following the festival.
  Fact 0.2: After Elizabeth is greeted by Hawk outside, Meliodas introduces her to a meat pie made with vegetables that looks like Hawk.
  Fact 0.3: The meat pie made with vegetables turns out to be unfit for human consumption.
  Fact 0.4: Meliodas has the idea of using pork as the main ingredient.
  Fact 0.5: Using pork as the main ingredient causes Hawk to make haste out of the bar.
  Fact 0.6: Meliodas comes up with a contest for his friends.
  Fact 0.7: In the contest, Meliodas will obey any order for a day from whoever catches Hawk first.
  Fact 0.8: Hawk manages to outsmart the contestants due to his small size.
  Fact 0.9: Hawk returns to the Boar Hat and eats food scraps made by Ban.
  Fact 0.10: Ban is declared the winner as a result.
  Fact 0.11: At night, Meliodas takes Elizabeth back to the palace.
  Fact 0.12: Meliodas assures Elizabeth that his purpose in life is to protect her.
  Fact 0.13: Meliodas then goes to the catacombs to see the Horn of Cernunnos.
  Fact 0.14: When the goddess vessel taunts Meliodas for his reputation as a demon, Meliodas destroys the Horn of Cernunnos in retaliation.
  Fact 0.15: The episode 'Our Fighting Festival' (Japanese title 'Futari no Ken'ka Matsuri', 二 人 の 喧嘩 祭り) first aired on September 4, 2016.
  Fact 0.16: Ban walks up to a mountaintop to finally settle matters with Meliodas.
  Fact 0.17: Ban is surprised to see Elizabeth and Hawk as well as the other members of the Seven Deadly Sins and the Holy Knights as part of an audience.
  Fact 0.18: Merlin casts a gigantic cubic barrier around Meliodas and Ban so the spectators can safely watch the fight.
  Fact 0.19: King and Hauser have their own fight to win the attention of Diane, meanwhile.
  Fact 0.20: Ban initiates a brutal attack on Meliodas that somehow weakens most of the spectators.
  Fact 0.21: The fight ends in a stalemate after the two of them exchange rapid punches that cause the barrier to break.
  Fact 0.22: Meliodas and Ban remain friends and head back down to the Boar Hat for a drink.
  Fact 0.23: The episode 'In Pursuit of First Love' (Japanese title 'Hatsukoi o Oikakete', 初恋 を 追いかけ て) first aired on September 11, 2016.
  Fact 0.24: King still hesitates to confess his feelings for Diane.
  Fact 0.25: Diane leaves the Boar Hat and heads to a construction site in town.
  Fact 0.26: Diane recalls the devastating collateral damage she caused due to her immense size.
  Fact 0.27: The construction workers at the site apologize to Diane and thank her for saving the kingdom.
  Fact 0.28: When King comes to see Diane, the two decide to play a game of tag across town.
  Fact 0.29: Outside a church, a wedding ceremony is interrupted by a demon.
  Fact 0.30: Diane and King combine their skills in defeating the demon.
  Fact 0.31: King asks Diane if she regained her memories, and she answers him with a kiss on his cheek.
  Fact 0.32: The moment is ruined when a brick suddenly falls on King's head.
  Fact 0.33: King later wakes up next to Diane, having forgotten what they did during the day.
  Fact 0.34: The episode 'The Shape of Love' (Japanese title 'Ai no Katachi', 愛 の かたち) first aired on September 18, 2016.
  Fact 0.35: Merlin agrees to be a waitress at the Boar Hat.
  Fact 0.36: Before taking the waitress job, Merlin wants Meliodas and Gowther to follow Gilthunder to find out if he is being stalked by someone.
  Fact 0.37: Gilthunder, Hauser and Griamore are each hit with falling debris in the streets, unaware of the situation.
  Fact 0.38: Gowther is distracted by a boy named Pelliot, who asks him to retrieve a fat cat off a roof.
  Fact 0.39: Gowther reminds Pelliot of his mother, who recently passed away.
  Fact 0.40: Gowther transforms into Pelliot's mother for a moment.
  Fact 0.41: Gowther disappears when Pelliot's father finds Pelliot.
  Fact 0.42: Meliodas finds Gilthunder, Hauser and Griamore at a restaurant.
  Fact 0.43: Meliodas purposely spills coffee to expose Vivian.
  Fact 0.44: Vivian is revealed as the stalker.
  Fact 0.45: Merlin punishes Vivian by planting a cursed ring on her finger.
  Fact 0.46: The cursed ring will bring pain to Vivian each time the curse is activated.
  Fact 0.47: Merlin explains that the Ten Commandments are the highest class in the demon race.
  Fact 0.48: At the Boar Hat, Merlin quits her job as a waitress.
  Fact 0.49: Hawk and Gowther fill in for the waitress position.
  Fact 0.50: Hawk and Gowther filling in for the waitress position is much to the guys' dismay.

Retrieved items:
<item 1>
List of The Seven Deadly Sins episodes
It is realized that Dale was one of Hendrickson 's failed demon experiments . 16 `` The Legends , Provoked '' `` Karitaterareru Densetsu - tachi '' ( 駆り立て られる
伝説 たち ) February 1 , 2015 Gowther is welcomed by the others into their group , but he soon finds out the true secrets behind them by delving deep into their
memories and emotions . Meanwhile , Hendrickson and Helbram have the hilt of Meliodas 's sword in possession , in which they plan to use it as a fragment of a
ritual relic , known as the Coffin of Eternal Darkness , to resurrect the red demon race as a means to start a holy war . At the Boar Hat , Holy Knight Vivian
suddenly appears and kidnaps Elizabeth and Hawk . Diane throws Meliodas , Ban and Gowther toward the Kingdom of Liones , where the three are bombarded by an
army of Holy Knights on the north side of the castle .
</item 1>

<item 2>
List of The Seven Deadly Sins episodes
14 `` A Reader of Books '' `` Hon o Yomu Hito '' ( 本 を 読む ひと ) January 18 , 2015 News about several sightings of an Armor Giant somewhere in Ordan Forest
reaches Helbram , who quickly sends Dawn Roar , an elite group of Holy Knights , to search for the Armor Giant . Meanwhile , Guila visits her younger brother
Zeal , and she later kills three thugs who bullied him for his spare change . Meliodas and his friends split up in the nearby village . Elizabeth and Hawk
encounter a boy named Alan , who is looking for some glue to repair armor . Upon hearing the low growls that echo in the forest , Meliodas , Ban and King find
the Armor Giant being surrounded by Dawn Roar , but Meliodas , Ban and King are no match against Dawn Roar . Just as it seems that the Armor Giant is about to
be slain , Alan intercepts and reveals himself as Gowther , the Goat 's Sin of Lust .
</item 2>

<item 3>
List of The Seven Deadly Sins episodes
Meanwhile , Great Holy Knight Hendrickson summons Jericho and Twigo underground , offering them blood from the corpse of a red demon as a means of enhancing
their strength . Twigo explodes after drinking the blood , but Jericho manages to survive the excruciating experience . 10 `` The Vaizel Fighting Festival '' ``
Baizeru Kenka Matsuri '' ( バイゼル 喧嘩 祭り ) December 7 , 2014 The fighting festival starts with an elimination round that results in eight contestants left standing
. The first round of the quarterfinals is Holy Knight Griamore versus `` Matrona '' , in which Matrona manages to break Griamore 's barrier and send him flying
. Matrona is revealed to be a shrunken Diane wearing Elizabeth 's clothing . Diane and Elizabeth previously encountered a giant mushroom called a Chicken -
Matango that shrunk them both when it felt threatened . The second round is Holy Knight Hauser versus three - time champion Taizoo , in which the former defeats
the latter with one punch .
</item 3>

<item 4>
List of The Seven Deadly Sins episodes
A second OVA composed of nine humorous shorts was shipped with the limited edition of the sixteenth volume of the manga , released on August 12 , 2015 . The
first The Seven Deadly Sins anime series was licensed for English release by Netflix as its second exclusive anime , following their acquisition of Knights of
Sidonia . All 24 episodes were released on November 1 , 2015 in both subtitled or English dub formats . The Seven Deadly Sins Revival of The Commandments first
opening theme song titled Howling is a collaboration between Flow and Granrodeo and first ending theme song is Beautiful performed by Anly . Contents ( hide ) 1
Episode list 1.1 The Seven Deadly Sins 1.2 The Seven Deadly Sins : Signs of Holy War 1.3 The Seven Deadly Sins : Revival of The Commandments 2 OVAs 3 References
Episode list ( edit ) The Seven Deadly Sins ( edit ) No .
</item 4>

<item 5>
List of The Seven Deadly Sins episodes
Vivian tries to trap Meliodas , Gilthunder , Margaret and Arthur in illusionary places , but Merlin , the Boar 's Sin of Gluttony , makes her first appearance
and breaks the spell . 21 `` The Looming Threat '' `` Ima , Soko ni Semaru Kyōi '' ( 今 、 そこ に せまる 脅威 ) March 8 , 2015 Hawk finds Ban in the catacombs underneath
the kingdom , where Ban makes contact with a goddess vessel through the Horn of Cernunnos . Ban is granted the request of resurrecting Elaine , but only with
the condition that he kills Meliodas . Meanwhile , Meliodas , Merlin , Gilthunder , Margaret and Arthur enter the chamber where Elizabeth and Bartra are being
held captive . Merlin takes Arthur and Bartra to Camelot to safety , but Hendrickson soon causes an explosion in the chamber , showing himself embedded in
darkness and causing terror in the town with the experimented Holy Knights turned into demons .
</item 5>

<item 6>
List of The Seven Deadly Sins episodes
Meanwhile , King and Hauser have their own fight to win the attention of Diane . Ban initiates a brutal attack on Meliodas that somehow weakens most of the
spectators . The fight ends in a stalemate after the two of them exchange rapid punches that cause the barrier to break . They remain friends and head back down
to the Boar Hat for a drink . `` In Pursuit of First Love '' `` Hatsukoi o Oikakete '' ( 初恋 を 追いかけ て ) September 11 , 2016 King still hesitates to confess his
feelings for Diane . Meanwhile , Diane leaves the Boar Hat and heads to a construction site in town , recalling the devastating collateral damage she caused due
to her immense size . The construction workers there apologize to her and thank her for saving the kingdom . When King comes to see Diane , the two decide to
play a game of tag across town . Outside a church , a wedding ceremony is interrupted by a demon , in which Diane and King combine their skills in defeating the
demon . King asks Diane if she regained her memories , and she answers him with a kiss on his cheek .
</item 6>

<item 7>
List of The Seven Deadly Sins episodes
Things return to normal inside the kingdom , and a festival is held at night . The next day , Elizabeth joins Meliodas , Hawk , Diane , Ban , King , Gowther and
Merlin in finding the last member of the Seven Deadly Sins , Escanor , the Lion 's Sin of Pride . The Seven Deadly Sins : signs of Holy war ( edit ) No . Title
Original air date `` The Dark Dream Begins '' `` Kuroki Yume no Hajimari '' ( 黒き 夢 の はじまり ) August 28 , 2016 Elizabeth Liones goes to the Boar Hat , where all
her friends are still asleep following the festival . After Elizabeth is greeted by Hawk outside , Meliodas introduces her to a meat pie made with vegetables
that looks like Hawk , but it turns out to be unfit for human consumption . Meliodas has the idea of using pork as the main ingredient , which causes Hawk to
make haste out of the bar . Meliodas comes up with a contest for his friends , in which he will obey any order for a day from whoever catches Hawk first .
</item 7>

<item 8>
List of The Seven Deadly Sins episodes
However , Arthur Pendragon , the young king of Camelot , has approached the south side of the castle with his own army . 17 `` The First Sacrifice '' `` Saisho
no Gisei '' ( 最初 の 犠牲 ) February 8 , 2015 Hendrickson tries to convince Arthur to leave the castle , but to no avail . Instead , Hendrickson escorts Arthur
inside the castle and suddenly engages him in battle . Meanwhile , Elizabeth and Hawk are teleported inside the dungeon of the castle , and they try to find a
way out . Meliodas , Ban and Gowther manage to defeat the army and sneak inside a town of the kingdom , while King summons his black hound Oslow to use its
mouth to teleport Diane inside the castle . However , Diane is attacked by Guila , Jericho , Gilthunder , Hauser , Helbram and Great Holy Knight Dreyfus . Diane
tries to defend herself , but Dreyfus overpowers her with an attack that sends her crashing onto the town streets .
</item 8>

<item 9>
Your Lie in April
! ! ( 2015 ) The Idolmaster Cinderella Girls : 2nd Season ( 2015 ) The Asterisk War ( 2015 -- 2016 ) Subete ga F ni Naru ( 2015 ) 2016 -- present Erased ( 2016
) Fairy Tail Zero ( 2016 ) Grimgar of Fantasy and Ash ( 2016 ) Ace Attorney ( 2016 -- 2018 ) B - Project : Kodō * Ambitious ( 2016 ) Qualidea Code ( 2016 ) The
Seven Deadly Sins : Signs of Holy War ( 2016 ) Persona 5 : The Animation - The Day Breakers - ( 2016 ) WWW. Working ! ! ( 2016 ) Uta no Prince - sama : Maji
Love Legend Star ( 2016 ) Occultic ;
</item 9>

<item 10>
List of The Seven Deadly Sins episodes
15 `` Unholy Knight '' `` Anhōryi Naito '' ( アンホーリィ ・ ナイト ) January 25 , 2015 Gowther gives the head of the Armor Giant to the members of Dawn Roar in order to
convince them to withdraw from battle . However , the demon breaks through the armor and attacks Meliodas , Ban , King and Gowther . Meliodas lowers his guard
upon noticing a Holy Knight merged into the demon 's chest , but Gowther uses his Sacred Treasure , the Twin Bow Herritt , to stop the demon . Meanwhile , Cain
visits Elizabeth in the Boar Hat , where he tells her the story of Liz , a knight from an enemy kingdom who became Meliodas 's past lover after he saved her
from being sentenced to death . Ban gouges out the heart of the Holy Knight , recognized as Guila 's father Dale , yet the demon still remains alive . Elizabeth
rushes to Meliodas and gives him Liz 's sword left behind , which allows him to destroy the demon . It is realized that Dale was one of Hendrickson 's failed
demon experiments .
</item 10>
=== END ITEM id=d7ea628d02a88626 ===
