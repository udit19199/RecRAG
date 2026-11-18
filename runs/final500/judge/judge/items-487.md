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

This file has 12 items.

=== ITEM id=2a7437e92bf3b83a ===
Gold sentences and facts:
Sentence 0: Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.
  Fact 0.0: Dangal's English title is "Wrestling competition".
  Fact 0.1: Dangal is a 2016 film.
  Fact 0.2: Dangal is an Indian film.
  Fact 0.3: Dangal is a Hindi-language film.
  Fact 0.4: Dangal is a biographical sports drama film.
  Fact 0.5: Dangal is loosely based on the story of Mahavir Singh Phogat.
  Fact 0.6: Mahavir Singh Phogat is an amateur wrestler.
  Fact 0.7: Mahavir Singh Phogat trains his daughters Geeta Phogat and Babita Kumari.
  Fact 0.8: Mahavir Singh Phogat trains his daughters to be world-class wrestlers.
Sentence 1: The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
  Fact 1.0: The Man from Snowy River II is a 1988 film.
  Fact 1.1: The Man from Snowy River II is an Australian drama film.
  Fact 1.2: The Man from Snowy River II is the sequel to the 1982 film "The Man from Snowy River".

Retrieved items:
<item 1>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 1>

<item 2>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 2>

<item 3>
The Man from Snowy River (poem)
"The Man from Snowy River" is a poem by Australian bush poet Banjo Paterson.

 It was first published in "The Bulletin", an Australian news magazine, on 26 April 1890, and was published by Angus & Robertson in October 1895, with other
poems by Paterson, in "The Man from Snowy River and Other Verses".
</item 3>

<item 4>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 4>

<item 5>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 5>

<item 6>
The Man from Snowy River II
The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
</item 6>

<item 7>
Graph query row:
film1: Dangal
year1: 2016
film2: The Man from Snowy River II
year2: 1988
came_first: The Man from Snowy River II
</item 7>

<item 8>
The Man from Snowy River (soundtrack)
"The Man from Snowy River is the original motion picture soundtrack from the 1982 film "The Man from Snowy River".
</item 8>

<item 9>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 9>

<item 10>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 10>
=== END ITEM id=2a7437e92bf3b83a ===

=== ITEM id=00babc289f569c5c ===
Gold sentences and facts:
Sentence 0:  The S70 was essentially a facelifted 850 saloon.
  Fact 0.0: The S70 was essentially a facelifted 850 saloon.
Sentence 1: The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.
  Fact 1.0: The Volvo 850 is a compact executive car.
  Fact 1.1: The Volvo 850 was produced by the Swedish manufacturer Volvo Cars.
  Fact 1.2: The Volvo 850 was produced from 1991 to 1996.
Sentence 2:  The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.
  Fact 2.0: The Volvo 850 was the first front-wheel drive vehicle from Volvo to be sold in North America.
  Fact 2.1: The Volvo 850 was the first all-wheel drive Volvo.

Retrieved items:
<item 1>
Volvo P2 platform
The Volvo P2 platform is a global full-size unibody automobile platform developed and produced by Volvo.

 It is designed for single wheelbases and is adaptable to front- or all wheel drive configurations.

 It was developed by the automaker before its 1999 acquisition by Ford Motor Company, and debuted with the 1998 Volvo S80.
</item 1>

<item 2>
Volvo SI6 engine
Volvo designed the SI6 ("short inline 6") straight-6 automobile engine for use in 2007 models.

 An evolution of the company's long-used straight-5 Volvo Modular engine, which itself is an evolution of the Volvo B6304 straight six engine, the SI6 can be
mounted transversely for front wheel drive or all wheel drive applications.

 Despite the added cylinder and displacement, the engine remains compact, and is in fact 1 mm shorter than the previous straight-5.

 The engine was initially offered in two displacements — a 3.0 L turbocharged version and a 3.2 L naturally aspirated version.

 Both offered variable cam timing, though only the turbo version varies both the intake and exhaust valves.

 On top of the variable cam timing used on the intake cam of the naturally aspirated engine it also had variable valve lift by using Cam Profile Switching
(CPS).
</item 2>

<item 3>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 3>

<item 4>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 4>

<item 5>
Volvo 164
The Volvo 164 is a 4-door, 6-cylinder luxury sedan unveiled by Volvo at the Paris Motor Show early in October 1968 and first sold as a 1969 model.

 46,008 164s were built before the car was superseded by the 264 in 1975.

 The 164 was Volvo's first venture into the luxury segment since the end of PV 60 production in 1950, and was the first six-cylinder Volvo since the PV800 last
produced in 1958.
</item 5>

<item 6>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 6>

<item 7>
Volvo 440/460
The Volvo 440 and 460 are versions of a small family car produced by the Swedish manufacturer Volvo between 1987 and 1996.

 The 440 was introduced in 1987, whilst the 460 followed in 1989.

 They were built at the NedCar factory in Born, The Netherlands.

 The 440 was a five door front wheel drive hatchback and the 460 a saloon.

 They shared many components with the already successful Volvo 480, including floorpan, front and rear suspension, engines, transmission and braking systems.
</item 7>

<item 8>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 8>

<item 9>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 9>

<item 10>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 10>
=== END ITEM id=00babc289f569c5c ===

=== ITEM id=0c162bbb50564b6d ===
Gold sentences and facts:
Sentence 0: The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran on Monday, February 8, 1993, killing all 133 people on board both aircraft involved—an Iran Air Tours Tupolev Tu-154M, registered EP-ITD, and an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
  Fact 0.0: The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran.
  Fact 0.1: The collision occurred on Monday, February 8, 1993.
  Fact 0.2: The collision killed all 133 people on board both aircraft involved.
  Fact 0.3: One aircraft involved was an Iran Air Tours Tupolev Tu-154M, registered EP-ITD.
  Fact 0.4: The other aircraft involved was an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
Sentence 1: Iran Air Tours is an airline based in Tehran, Iran.
  Fact 1.0: Iran Air Tours is an airline.
  Fact 1.1: Iran Air Tours is based in Tehran, Iran.

Retrieved items:
<item 1>
The entries cover major aviation accidents, airlines, airports, and safety topics. Notable crashes include Flight 2311 (23 dead), Pan Am 806 (97), the 1948
Northwood collision (39), Tehran 1993 (133), Charkhi Dadri 1996 (349), and United 736 (49); Flight 1549 and the 1963 Neva ditching had no fatalities. TCAS warns
of collision threats, while bird strikes can cause crashes,
</item 1>

<item 2>
Traffic collision avoidance system
A traffic collision avoidance system or traffic alert and collision avoidance system (both abbreviated as TCAS, and pronounced tee-kas) is an aircraft collision
avoidance system designed to reduce the incidence of mid-air collisions between aircraft.

 It monitors the airspace around an aircraft for other aircraft equipped with a corresponding active transponder, independent of air traffic control, and warns
pilots of the presence of other transponder-equipped aircraft which may present a threat of mid-air collision (MAC).

 It is a type of airborne collision avoidance system mandated by the International Civil Aviation Organization to be fitted to all aircraft with a maximum
take-off mass (MTOM) of over 5700 kg or authorized to carry more than 19 passengers.

 CFR 14, Ch I, part 135 requires that TCAS I is installed for aircraft with 10-30 passengers and TCAS II for aircraft with more than 30 passengers.
</item 2>

<item 3>
1993 Tehran mid-air collision
The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran on Monday, February 8, 1993, killing all 133 people on board both
aircraft involved—an Iran Air Tours Tupolev Tu-154M, registered EP-ITD, and an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
</item 3>

<item 4>
1948 Northwood mid-air collision
The 1948 Northwood mid-air collision took place on 4 July at 15:03 when a Douglas DC-6 of Scandinavian Airlines System (SAS) and an Avro York C.1 of the Royal
Air Force (RAF) collided in mid-air over Northwood in London, UK.

 All thirty-nine people aboard both aircraft were killed.

 It was SAS's first fatal aviation accident and was at the time the deadliest civilian aviation accident in the UK.

 It is still the deadliest mid-air collision in British history.
</item 4>

<item 5>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 5>

<item 6>
Iran Air Tours
Iran Air Tours is an airline based in Tehran, Iran.

 The airline mainly operates under the English name Iran Airtour (i.e. without the plural "s") as written in Persian (Persian: ایران ایر تور‎ ‎ ).

 This error is due to confusion between different pluralization rules in English and Persian.

 Iran Air Tours was established as a subsidiary of Iran Air and operates scheduled domestic services and international services in the Middle East as well as
Eastern Europe, as well as charter services.
</item 6>

<item 7>
1993 Auckland mid-air collision
The 1993 Auckland mid-air collision was an aircraft accident in New Zealand.

 It occurred on 26 November 1993, when two aircraft operated by Airwork, under contract to the New Zealand Police, collided and crashed in central Auckland.

 The mid-air collision of the Aérospatiale TwinStar helicopter and Piper Archer aeroplane resulted in the deaths of all four occupants – a civilian Airwork
pilot on each aircraft and two New Zealand Police officers on the helicopter.

 The accident occurred in daylight with excellent visibility, in uncontrolled airspace (class G), with both aircraft flying under visual flight rules.

 Both the helicopter and aeroplane were operated by Airwork (NZ), and working under contract to the New Zealand Police at the time of the accident.
</item 7>

<item 8>
Charkhi Dadri mid-air collision
The Charkhi Dadri mid-air collision occurred on 12 November 1996 over the village of Charkhi Dadri, to the west of New Delhi, India.

 The aircraft involved were a Saudi Arabian Airlines Boeing 747-100B en route from Delhi to Dhahran, Saudi Arabia, and a Kazakhstan Airlines Ilyushin Il-76 en
route from Chimkent, Kazakhstan, to Delhi.

 The crash killed all 349 people on board both planes, making it the world's deadliest mid-air collision, the deadliest aviation accident to occur in India, and
the third-deadliest aircraft accident in the history of aviation, behind only the Tenerife airport disaster and Japan Airlines Flight 123.
</item 8>

<item 9>
1955 Cincinnati mid-air collision
The 1955 Cincinnati mid-air collision occurred on January 12, 1955, when a Trans World Airlines Martin 2-0-2 on takeoff from Boone County Airport (now the
Cincinnati/Northern Kentucky International Airport) collided in mid-air with a privately owned Douglas DC-3 that had entered the airport's control space without
proper clearance.

 There were no survivors.
</item 9>

<item 10>
1963 Aeroflot Tupolev Tu-124 Neva river ditching
The 1963 Aeroflot Tupolev Tu-124 Neva river ditching was a water landing by a Tupolev Tu-124 of the Soviet state airline "Aeroflot" (Moscow division).

 The aircraft took off from Tallinn-Ülemiste Airport (TLL) at 08:55 on 21 August with 45 passengers and 7 crew on board.

 The aircraft (registration number SSSR-45021) was built in 1962 and was scheduled to fly to Moscow–Vnukovo (VKO) under the command of 27-year-old captain
Victor Mostovoy.

 After takeoff the nose gear did not retract.

 Ground control diverted the flight to Leningrad (LED) – because of fog at Tallinn.
</item 10>
=== END ITEM id=0c162bbb50564b6d ===

=== ITEM id=5248ef1ada77368e ===
Gold sentences and facts:
Sentence 0: Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct "Final Destination 5" six years later, and filmed in the IMAX 3D format.
  Fact 0.0: Aliens of the Deep is a 2005 documentary film.
  Fact 0.1: Aliens of the Deep was directed in part by James Cameron.
  Fact 0.2: James Cameron directed Aliens of the Deep alongside fellow cameraman and friend Steven Quale.
  Fact 0.3: Steven Quale would go on to direct Final Destination 5 six years later.
  Fact 0.4: Aliens of the Deep was filmed in the IMAX 3D format.
Sentence 1:  Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of the planet's more unusual forms of life.
  Fact 1.0: James Cameron teams with NASA scientists to explore the Mid-Ocean Ridges.
  Fact 1.1: The Mid-Ocean Ridges are submerged chains of mountains.
  Fact 1.2: The Mid-Ocean Ridges are in the Atlantic and Pacific oceans.
  Fact 1.3: The Mid-Ocean Ridges are home to some of the planet's more unusual forms of life.
Sentence 2: Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.
  Fact 2.0: Poetry in Motion is a 1982 Canadian documentary film.
  Fact 2.1: Poetry in Motion was directed by Ron Mann.
  Fact 2.2: Poetry in Motion features contemporary North American poetry.
  Fact 2.3: Poetry in Motion features contemporary North American music.
Sentence 3:  Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.
  Fact 3.0: Poetry in Motion features some Black Mountain poets.
  Fact 3.1: Poetry in Motion features some Beat poets.
  Fact 3.2: Poetry in Motion features some minimalist poets.
  Fact 3.3: Poetry in Motion features some avant-garde poets.
Sentence 4:  It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.
  Fact 4.0: Poetry in Motion was released in theaters.
  Fact 4.1: Poetry in Motion was later distributed on VHS.
  Fact 4.2: Poetry in Motion was later distributed on LaserDisc.
  Fact 4.3: Poetry in Motion was later distributed on DVD.
Sentence 5:  An extended CD-ROM version was also released.
  Fact 5.0: An extended CD-ROM version of Poetry in Motion was also released.

Retrieved items:
<item 1>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 1>

<item 2>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 2>

<item 3>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 3>

<item 4>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 4>

<item 5>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 5>

<item 6>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 6>

<item 7>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 7>

<item 8>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 8>

<item 9>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 9>

<item 10>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 10>
=== END ITEM id=5248ef1ada77368e ===

=== ITEM id=a46833e371c6c298 ===
Gold sentences and facts:
Sentence 0:  She was also a member of a secret society composed exclusively of Cheyenne women.
  Fact 0.0: The woman in question was also a member of a secret society.
  Fact 0.1: The secret society was composed exclusively of Cheyenne women.
Sentence 1: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.
  Fact 1.0: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance.
  Fact 1.1: The other central institution of traditional Cheyenne Indian tribal governance is the Council of Forty-four.

Retrieved items:
<item 1>
Chief Dull Knife College
Chief Dull Knife College (originally Dull Knife Memorial College) is a small, open-admission, Native American tribal community college and land grant
institution.

 Located on the Northern Cheyenne Indian Reservation, in the city of Lame Deer, it has a current enrollment of about 141 students.

 On average, more than half of its graduates move on to four-year colleges.
</item 1>

<item 2>
Council of Forty-four
The Council of Forty-four was one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the military societies such
as the Dog Soldiers.

 The influence of the Council of Forty-four waned in the face of internal conflict among the Cheyenne about Cheyenne policy toward encroaching white settlers on
the Great Plains, and was dealt a severe blow by the Sand Creek Massacre.
</item 2>

<item 3>
Ehyophsta
Ehyophsta (Cheyenne for "Yellow-Haired Woman") was a Cheyenne woman.

 She was the daughter of Stands-in-the-Timber who died in 1849, and she was the niece of Bad Faced Bull.

 She fought in the Battle of Beecher Island in 1868, and also fought the Shoshone that same year, where she counted coup against one enemy and killed another.

 She fought the Shoshone again in 1869.

 She was also a member of a secret society composed exclusively of Cheyenne women.

 She died in 1915.
</item 3>

<item 4>
Northern Cheyenne Indian Reservation
The Northern Cheyenne Indian Reservation (Tsėhéstáno in Cheyenne, formerly named the Tongue River Indian Reservation) is home of the federally recognized
Northern Cheyenne Tribe.

 Located in southeastern Montana, the reservation is approximately 444,000 acres in size and home to approximately 5,000 Cheyenne people.

 The tribal and government headquarters are in Lame Deer, which is also the home of the annual Northern Cheyenne Pow wow.

 The reservation is bounded on the east by the Tongue River and on the west by the Crow Reservation.

 There are small parcels of non-contiguous off-reservation trust lands in Meade County, South Dakota, northeast of the city of Sturgis.

 Its timbered ridges that extend into northwestern South Dakota are part of Custer National Forest and it is approximately 40 mi east of the site of the 1876
Battle of the Greasy Grass (known to most Americans as the Battle of the Little Bighorn).
</item 4>

<item 5>
Eugene Little Coyote
Eugene Little Coyote was the president of the Northern Cheyenne Indian Reservation from 2004 to 2007.

 He was elected in November 2004, defeating the incumbent president, Geri Small.

 However, after a conflict with reservation vice president Rick Wolfname that began in July 2007 escalated, the Northern Cheyenne tribal council declared that
it would oust Little Coyote as the tribe's president.

 Little Coyote was forcibly removed from office and arrested for trespassing in the tribal office building on December 28, 2007.

 It is worth noting that the move by the tribal council to remove Little Coyote from office was deemed unconstitutional and void by the Northern Cheyenne
Constitutional Court.

 The Constitutional Court was however overruled by Bureau of Indian Affairs' regional director Ed Parisian, and later the Assistant Secretary of Indian Affairs
Carl J. Artman.

 Their involvement in this matter set a historical precedence of the BIA directly meddling with tribal governments.
</item 5>

<item 6>
Lean Bear
Lean Bear (Cheyenne name Awoninahku, 1813–1864), alternatively translated as Starving Bear, was a Cheyenne peace chief.

 He was a member of the Council of Forty-four, a tribal governance devoted to maintaining peace with encroaching United States settlers.

 Lean Bear's most notable peace deals include the Treaty of Fort Wise and a meeting with US President Abraham Lincoln.

 His work towards peace between his people and the American settlers in the Southern Plains were cut short when he was murdered by the 1st Colorado Cavalry
Regiment and violent retaliations ensued.
</item 6>

<item 7>
Cheyenne
The Cheyenne ( ) are one of the Indigenous peoples of the Great Plains and their language is of the Algonquian language family.

 The Cheyenne comprise two Native American tribes, the Só'taeo'o or Só'taétaneo'o (more commonly spelled as Suhtai or Sutaio) and the Tsétsêhéstâhese (also
spelled Tsitsistas).

 These tribes merged in the early 19th century.

 Today, the Cheyenne people are split into two federally recognized Nations: the Southern Cheyenne, who are enrolled in the Cheyenne and Arapaho Tribes in
Oklahoma, and the Northern Cheyenne, who are enrolled in the Northern Cheyenne Tribe of the Northern Cheyenne Indian Reservation in Montana.
</item 7>

<item 8>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 8>

<item 9>
Cheyenne military societies
Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.

 While council chiefs are responsible for overall governance of individual bands and the tribe as a whole, the headmen of military societies are in charge of
maintaining discipline within the tribe, overseeing tribal hunts and ceremonies, and providing military leadership.

 Historically, council chiefs selected which of the six military societies would assume these duties; after a period of time on-duty, the chiefs would select a
different society to take up the duties.
</item 9>

<item 10>
Concho Indian Boarding School
Concho Indian Boarding School (also known as the Cheyenne-Arapaho Boarding School at Concho or Concho Indian School and home to the Concho Demonstration School)
was a boarding school for members of the Cheyenne and Arapaho Tribes and later opened to other Native American students.

 It existed from 1909 to 1983.

 It was located in central Oklahoma, approximately 1 mile south of Concho, Oklahoma and 4 miles north of El Reno, Oklahoma.

 The name of the town and school is the Spanish word for "shell" and was named for the Indian agent, Charles E. Shell.
</item 10>
=== END ITEM id=a46833e371c6c298 ===

=== ITEM id=6db2477c31bc27f0 ===
Gold sentences and facts:
Sentence 0: Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at Naval Air Station Whidbey Island, Washington.
  Fact 0.0: Electronic Attack Squadron 138 (VAQ-138) is also known as the "Yellow Jackets".
  Fact 0.1: Electronic Attack Squadron 138 is an expeditionary EA-18G Growler squadron of the United States Navy.
  Fact 0.2: Electronic Attack Squadron 138 is based at Naval Air Station Whidbey Island, Washington.
Sentence 1: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.
  Fact 1.0: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft.
  Fact 1.1: The Boeing EA-18G Growler is a specialized version of the two-seat F/A-18F Super Hornet.

Retrieved items:
<item 1>
Boeing EA-18G Growler
The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.

 The EA-18G replaced the Northrop Grumman EA-6B Prowlers in service with the United States Navy.

 The Growler's electronic warfare capability is primarily provided by Northrop Grumman.

 The EA-18G began production in 2007 and entered operational service with the US Navy in late 2009.

 Australia has also purchased twelve EA-18Gs, which entered service with the Royal Australian Air Force in 2017.
</item 1>

<item 2>
Boeing F/A-18E/F Super Hornet
The Boeing F/A-18E and F/A-18F Super Hornet are twin-engine carrier-capable multirole fighter aircraft variants based on the McDonnell Douglas F/A-18 Hornet.

 The F/A-18E single-seat and F/A-18F tandem-seat variants are larger and more advanced derivatives of the F/A-18C and D Hornet.

 The Super Hornet has an internal 20 mm M61 rotary cannon and can carry air-to-air missiles and air-to-surface weapons.

 Additional fuel can be carried in up to five external fuel tanks and the aircraft can be configured as an airborne tanker by adding an external air refueling
system.
</item 2>

<item 3>
VAQ-209
Electronic Attack Squadron 209 (VAQ-209) is a United States Navy Reserve electronic attack squadron.

 Known as the "Star Warriors", the squadron flies the EA-18G Growler carrier-based electronic warfare jet aircraft.

 Based at Naval Air Station Whidbey Island, WA, it is assigned to the Tactical Support Wing.
</item 3>

<item 4>
VFA-154
Strike Fighter Squadron 154 (VFA-154), also known as the "Black Knights", is a United States Navy strike fighter squadron stationed at Naval Air Station
Lemoore.

 The Black Knights are an operational fleet squadron flying the F/A-18F Super Hornet.

 They are currently attached to Carrier Air Wing Eleven and deployed aboard the aircraft carrier USS "Nimitz" .

 Their tailcode is NH and their callsign is "Knight".
</item 4>

<item 5>
VAQ-135
Electronic Attack Squadron 135 (VAQ-135), known as the "Black Ravens", is a United States Navy electronic attack squadron that currently operates the EA-18G
Growler carrier-based electronic warfare jet aircraft.

 The squadron is permanently stationed at Naval Air Station Whidbey Island with a radio callsign of ""Thunder"".
</item 5>

<item 6>
No. 1 Squadron RAAF
No. 1 Squadron is a Royal Australian Air Force (RAAF) squadron headquartered at RAAF Base Amberley, Queensland.

 Controlled by No. 82 Wing, it is equipped with Boeing F/A-18F Super Hornet multi-role fighters.

 The squadron was formed under the Australian Flying Corps in 1916 and saw action in the Sinai and Palestine Campaigns during World War I.

 It flew obsolete Royal Aircraft Factory B.E.2s, B.E.12s, Martinsyde G.100s and G.102s, as well as Airco DH.6s, Bristol Scouts and Nieuport 17s, before
re-equipping with the R.E.8 in October 1917 and finally the Bristol Fighter in December.

 Its commanding officer in 1917–18 was Major Richard Williams, later known as the "Father of the RAAF".

 Disbanded in 1919, No. 1 Squadron was re-formed on paper as part of the RAAF in 1922, and re-established as an operational unit three years later.
</item 6>

<item 7>
No. 82 Wing RAAF
No. 82 Wing is the strike and reconnaissance wing of the Royal Australian Air Force (RAAF).

 It is headquartered at RAAF Base Amberley, Queensland.

 Coming under the control of Air Combat Group, the wing operates F/A-18F Super Hornet multirole fighters and Pilatus PC-9 forward air control aircraft.

 Its units include Nos. 1 and 6 Squadrons, operating the Super Hornet, and No. 4 Squadron, operating the PC-9.
</item 7>

<item 8>
VAQ-138
Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at
Naval Air Station Whidbey Island, Washington.

 Their radio callsign is "RAMPAGE".
</item 8>

<item 9>
Type 002 aircraft carrier
The Type 002 aircraft carrier is a second generation Chinese aircraft carrier design.

 The ship will be the first Chinese aircraft carrier to be equipped with a catapult.

 According to the latest report, the construction of this carrier has been rescheduled for a decision on the catapults, where the latest competition results
suggested that steam catapults will be used in the Type 002 aircraft carrier.

 The addition of aircraft catapult will give China the ability to launch various fixed wing support aircraft like airborne early warning (AEW) and carrier
onboard delivery aircraft.

 Fighters can also be launched at full combat load and reach its full potential unlike current Type 001 aircraft carriers operated by Chinese navy.
</item 9>

<item 10>
VFA-2
Strike Fighter Squadron 2 (VFA-2) also known as the "Bounty Hunters" is a United States Navy F/A-18F Super Hornet strike fighter squadron based at Naval Air
Station Lemoore, California.

 Their tail code is NE and their callsign is "Bullet".

 They are attached to Carrier Air Wing 2 (CVW-2), a composite unit made up of a wide array of aircraft performing a variety of combat and support missions that
deploy aboard the "Carl Vinson" .
</item 10>
=== END ITEM id=6db2477c31bc27f0 ===

=== ITEM id=31b17127ebd5ca5c ===
Gold sentences and facts:
Sentence 0: John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).
  Fact 0.0: John H. Garabedian is an American radio personality.
  Fact 0.1: John H. Garabedian is an American disc jockey.
  Fact 0.2: John H. Garabedian was born on December 20, 1941.
  Fact 0.3: John H. Garabedian is age 75.
Sentence 1: Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
  Fact 1.0: Charles A. Agemian was born July 22, 1909.
  Fact 1.1: Charles A. Agemian's life ended March 30, 1996.
  Fact 1.2: Charles A. Agemian was an American banker.
  Fact 1.3: Charles A. Agemian took early retirement from Chase Manhattan Bank.
  Fact 1.4: Charles A. Agemian was executive vice president of operations at Chase Manhattan Bank.
  Fact 1.5: Charles A. Agemian became chairman of the Hackensack Trust Co.
  Fact 1.6: Charles A. Agemian became chief executive officer of the Hackensack Trust Co.
  Fact 1.7: The Hackensack Trust Co. was later renamed Garden State National Bank.

Retrieved items:
<item 1>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 1>

<item 2>
Don Sherwood (DJ)
Don Sherwood (September 7, 1925 – November 6, 1983) was an American radio personality.

 He was a San Francisco, California, disc jockey during the 1950s and 1960s.

 Billed as "The World's Greatest Disc Jockey," Sherwood spent most of his career hosting a 6-9 a.m. weekday program on KSFO in San Francisco (560 kHz, 5000
watts), which was then owned by the singing cowboy actor Gene Autry.
</item 2>

<item 3>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 3>

<item 4>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 4>

<item 5>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 5>

<item 6>
Radio personality
A radio personality (American English) or radio presenter (British English), commonly referred to as a "disc jockey" or "DJ" for short, is a person who has an
on-air position in radio broadcasting.

 A radio personality that hosts a radio show is also known as a radio host, and in India and Pakistan as a radio jockey.

 Radio personalities who introduce and play individual selections of recorded music are known as disc jockeys.

 The term has evolved to also describe a person who mixes a continuous flow of recorded music in real time.

 Broadcast radio personalities may include talk radio hosts, AM/FM radio show hosts, and satellite radio program hosts.

 Notable radio personalities include pop music radio hosts Martin Block, Alan Freed, Dick Clark, Delilah Luke, Ameen Sayani, Wolfman Jack, and Casey Kasem,
shock jocks such as Don Imus and Howard Stern, as well as sports talk hosts such as Mike Francesa and political talk hosts such as Rush Limbaugh.
</item 6>

<item 7>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 7>

<item 8>
M. G. Kelly
Gary Sinclair (born 1952 in Ada, Oklahoma), known professionally as Michael Gary “M.G.” Kelly and Machine Gun Kelly, is an American actor, disc jockey and radio
personality.

 In addition to hosting several radio programs over the years, Kelly has held several acting roles as a disc jockey; also, he has served as an offstage
announcer on two game shows.
</item 8>

<item 9>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 9>

<item 10>
Big Wilson
Big Wilson (born Malcolm John Wilson, Jr.; October 3, 1924 – October 5, 1989) was an American radio personality.

 He worked as a disc jockey at WNBC AM in New York City from the early 1960s until 1974 and moved to Miami in 1975 where he worked for WIOD and WCIX-TV.

 He was one of the last two hosts of the NBC network program Monitor.

 Prior to moving to NYC, Wilson and his wife Jody lived in a riverfront home in Rocky River, Ohio.

 He commuted to Cleveland where he was the leading radio personality for some years.

 Tim Conway and other well-known persons visited their home.
</item 10>
=== END ITEM id=31b17127ebd5ca5c ===

=== ITEM id=8646e48abc927563 ===
Gold sentences and facts:
Sentence 0: Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.
  Fact 0.0: Bluebeard's Castle is a one-act opera.
  Fact 0.1: Bluebeard's Castle is by Hungarian composer Béla Bartók.
  Fact 0.2: The Hungarian title of Bluebeard's Castle is A kékszakállú herceg vára.
  Fact 0.3: Literally, the Hungarian title means The Blue-Bearded Duke's Castle.
Sentence 1: Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.
  Fact 1.0: Agrippina (HWV 6) is an opera seria.
  Fact 1.1: Agrippina (HWV 6) has three acts.
  Fact 1.2: Agrippina (HWV 6) is by George Frideric Handel.
  Fact 1.3: Agrippina (HWV 6) has a libretto by Cardinal Vincenzo Grimani.

Retrieved items:
<item 1>
Arlecchino (opera)
Arlecchino, "oder Die Fenster" ("Harlequin, or The Windows", is a one-act opera with spoken dialog by Ferruccio Busoni, with a libretto in German, composed in
1913.

 He completed the music for the opera while living in Zurich in 1916.

 It is a number opera written in neo-classical style and includes ironic allusions to operatic conventions and situations typical of the late 18th and early
19th centuries.

 It even includes a parody of a duel.
</item 1>

<item 2>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 2>

<item 3>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 3>

<item 4>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 4>

<item 5>
Bluebeard's Castle
Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.

 The libretto was written by Béla Balázs, a poet and friend of the composer, and is written in Hungarian, based on the French literary tale "La Barbe bleue" by
Charles Perrault.

 The opera lasts only a little over an hour and there are only two singing characters onstage: Bluebeard ("Kékszakállú "), and his new wife Judith ("Judit " );
the two have just eloped and Judith is coming home to Bluebeard's castle for the first time.
</item 5>

<item 6>
At the Boar's Head
At the Boar's Head is an opera in one act by the English composer Gustav Holst, his op. 42.

 Holst himself described the work as "A Musical Interlude in One Act".

 The libretto, by the composer himself, is based on Shakespeare's "Henry IV, Part 1" and "Henry IV, Part 2".
</item 6>

<item 7>
The Padlock
The Padlock is a two-act 'afterpiece' opera by Charles Dibdin.

 The text was by Isaac Bickerstaffe.

 It debuted in 1768 at the Drury Lane Theatre in London as a companion piece to "The Earl of Warwick".

 It partnered other plays before a run of six performances in tandem with "The Fatal Discovery" by John Home.

 "The Padlock" was a success, largely due to Dibdin's portrayal of Mungo, a blackface caricature of a black servant from the West Indies.

 The company took the production to the United States the next year, where a portrayal by Lewis Hallam, Jr. as Mungo met with even greater accolades.

 The libretto was first published in London in 1768 (?)

 and in Dublin in 1775.

 The play remained in regular circulation in the U.S. as late as 1843.

 It was revived by the Old Vic Company in London and on tour in the UK in 1979 in a new orchestration by Don Fraser and played in a double-bill with Garrick's
"Miss In Her Teens".
</item 7>

<item 8>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 8>

<item 9>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 9>

<item 10>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 10>
=== END ITEM id=8646e48abc927563 ===

=== ITEM id=e30b93434a644b5d ===
Gold sentences and facts:
Sentence 0: Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.
  Fact 0.0: Heavenly Creatures is a 1994 New Zealand psychological drama.
  Fact 0.1: Heavenly Creatures was directed by Peter Jackson.
  Fact 0.2: Peter Jackson co-wrote the screenplay with his partner, Fran Walsh.
  Fact 0.3: Heavenly Creatures is about the notorious 1954 Parker-Hulme murder case.
  Fact 0.4: The Parker-Hulme murder case took place in Christchurch, New Zealand.
Sentence 1:  The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.
  Fact 1.0: The main premise of Heavenly Creatures deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme.
  Fact 1.1: Pauline Parker and Juliet Hulme murder Pauline Parker's mother.
Sentence 2: Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk series.
  Fact 2.0: Anne Perry was born on 28 October 1938 as Juliet Marion Hulme.
  Fact 2.1: Anne Perry is an English author of historical detective fiction.
  Fact 2.2: Anne Perry is best known for her Thomas Pitt and William Monk series.

Retrieved items:
<item 1>
Hulder
A hulder is a seductive forest creature found in Scandinavian folklore.

 (Her name derives from a root meaning "covered" or "secret".)

 In Norwegian folklore, she is known as huldra ("the [archetypal] hulder", though folklore presupposes that there is an entire Hulder race and not just a single
individual).

 She is known as the skogsrå "forest spirit" or Tallemaja "pine tree Mary" in Swedish folklore, and ulda in Sámi folklore.

 Her name suggests that she is originally the same being as the völva divine figure Huld and the German Holda.
</item 1>

<item 2>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 2>

<item 3>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 3>

<item 4>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 4>

<item 5>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 5>

<item 6>
Julia (1977 film)
Julia is a 1977 American drama film directed by Fred Zinnemann, from a screenplay by Alvin Sargent.

 It is based on a chapter from Lillian Hellman's book "Pentimento" about the author's relationship with a lifelong friend, "Julia," who fought against the Nazis
in the years prior to World War II.

 Hellman said the story was true, but critics have challenged its accuracy.

 The film in DeLuxe Color was produced by Richard Roth, with Julien Derode as executive producer and Tom Pevsner as associate producer.
</item 6>

<item 7>
Peter Elliott (actor)
Peter Elliott is a leading New Zealand actor.

 He has appeared in numerous television shows including "Shortland Street", "Gloss" and .

 He has also appeared in several movies including "Heavenly Creatures".

 Peter has a daughter Lucy Elliott who is an actress, playing character Dayna Jenkins on "Shortland Street" in 2013.
</item 7>

<item 8>
Heavenly Creatures
Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the
notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.

 The film features Melanie Lynskey and Kate Winslet in their screen debuts with supporting roles by Sarah Peirse, Diana Kent, Clive Merrison, and Simon
O'Connor.

 The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.

 The events of the film cover the period from their meeting in 1952 to the murder in 1954.
</item 8>

<item 9>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 9>

<item 10>
Parker–Hulme murder case
The Parker–Hulme murder case began in the city of Christchurch, New Zealand, on 22 June 1954, when Honora Rieper (also known as Honora Parker, her legal name)
was killed by her teenaged daughter, Pauline Parker, and Pauline's close friend Juliet Hulme (later known as Anne Perry).

 Parker was 16 at the time, while Hulme was 15.
</item 10>
=== END ITEM id=e30b93434a644b5d ===

=== ITEM id=4364aaf4a7bec2e3 ===
Gold sentences and facts:
Sentence 0: Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.
  Fact 0.0: Saddle Rock Elementary School is also called Saddle Rock.
  Fact 0.1: Saddle Rock Elementary School is an elementary school.
  Fact 0.2: Saddle Rock Elementary School comprises grades Kindergarten through 5.
Sentence 1:  It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.
  Fact 1.0: Saddle Rock Elementary School is a public school.
  Fact 1.1: Saddle Rock Elementary School is located in Great Neck, New York, USA.
  Fact 1.2: Saddle Rock Elementary School is part of the Great Neck School District.
Sentence 2: The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.
  Fact 2.0: The Great Neck School District is a comprehensive community public school district.
  Fact 2.1: The Great Neck School District primarily serves students in Great Neck, New York.
Sentence 3:  It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
  Fact 3.0: The Great Neck School District is Union Free School District Number 7.
  Fact 3.1: The Great Neck School District is in the Town of North Hempstead.
  Fact 3.2: The Great Neck School District is in Nassau County, New York, United States.
  Fact 3.3: The Great Neck School District is on Long Island.

Retrieved items:
<item 1>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 1>

<item 2>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 2>

<item 3>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 3>

<item 4>
Saddle Rock Elementary School
Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.

 It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.

 The school has been recognized by the Blue Ribbon Schools Program.
</item 4>

<item 5>
East Lyme High School
East Lyme High School is a high school located in the Flanders Village region of East Lyme, Connecticut.

 It is operated by East Lyme Public Schools.

 The mascot is Sven the Viking.

 Students from the town of Salem, Connecticut in grades 9 through 12 attend high school in East Lyme (as they have no high school of their own); this will be
the case until at least 2016 when the current co-op agreement between the two towns expires.
</item 5>

<item 6>
Ridgefield Park station
Ridgefield Park Station, also known as West Shore Station, was railroad station in Ridgefield Park, New Jersey at the foot of Mount Vernon Street served by the
New York, Susquehanna and Western Railroad (NYSW) and the West Shore Railroad, a division of New York Central (NYCRR) The New York, Ontario and Western Railway
(NYO&W) had running rights along the West Shore and sometimes stopped at Ridgefield Park.

 First opened in 1883 it was one of three passenger stations in the village, the others being the Little Ferry Station to the south and Westview Station to the
north.

 The station house, built at a cost $100,000 opened in 1927.

 Southbound service crossed Overpeck Creek and continued to terminals on the Hudson River waterfront where there was connecting ferry service across the Hudson
River to Manhattan.

 Northbound near Bogota the parallel NYSW and West Shore lines diverge and continue into northern New Jersey, Pennsylviania, and upstate New York.

 Passenger service ended in 1966.
</item 6>

<item 7>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 7>

<item 8>
Village School (Great Neck, New York)
Village School (also known as VS) is an American long-established public alternative school.

 It is a member of Coalition of Essential Schools and is located in the village of Great Neck, New York, serving students in grades 8 through 12.

 Village School is one of three high schools in the Great Neck School District, which includes Great Neck North High School and Great Neck South High School.

 Village School offers its 39 students an outdoor education program, college preparatory program, and inclusion of students with disabilities.
</item 8>

<item 9>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 9>

<item 10>
Monmouth Heights at Marlboro, New Jersey
Monmouth Heights at Marlboro is an unincorporated community located within Marlboro Township in Monmouth County, New Jersey, United States.

 The housing development was created in the 1960s by Levitt & Sons though some protests occurred by long-time residents of Marlboro Township at the time of its
creation.

 The community is located along New Jersey Route 79 across the highway from Marlboro High School.
</item 10>
=== END ITEM id=4364aaf4a7bec2e3 ===

=== ITEM id=e35cdf2251637644 ===
Gold sentences and facts:
Sentence 0: David Michael Draiman (born March 13, 1973) is an American songwriter and the vocalist for the band Disturbed as well as for the band Device.
  Fact 0.0: David Michael Draiman was born on March 13, 1973.
  Fact 0.1: David Michael Draiman is an American songwriter.
  Fact 0.2: David Michael Draiman is the vocalist for the band Disturbed.
  Fact 0.3: David Michael Draiman is the vocalist for the band Device.
Sentence 1: Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and founding members of the rock band Chicago.
  Fact 1.0: Terry Alan Kath lived January 31, 1946 to January 23, 1978.
  Fact 1.1: Terry Alan Kath was an American musician.
  Fact 1.2: Terry Alan Kath was an American songwriter.
  Fact 1.3: Terry Alan Kath was best known as the original guitarist of the rock band Chicago.
  Fact 1.4: Terry Alan Kath was one of the lead singers of the rock band Chicago.
  Fact 1.5: Terry Alan Kath was a founding member of the rock band Chicago.

Retrieved items:
<item 1>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 1>

<item 2>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 2>

<item 3>
Steve Alvord
Steven Lee Alvord (born October 2, 1964 in Bellingham, Washington) is a former professional American football defensive tackle and defensive end in the National
Football League and World League of American Football.

 In his four-year pro career he played for the St. Louis/Phoenix Cardinals of the NFL, and the Barcelona Dragons of the WLAF.

 Alvord played college football at Washington.

He has a cool daughter, happy birthday Katie.
</item 3>

<item 4>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 4>

<item 5>
Karlfried Graf Dürckheim
Karl Friedrich Alfred Heinrich Ferdinand Maria Graf Eckbrecht von Dürckheim-Montmartin (24 October 1896 – 28 December 1988) was a German diplomat,
psychotherapist and Zen Master.

 A veteran of World War I, he was introduced to Zen Buddhism early in life.

 After obtaining a doctorate in psychology, he became an avid supporter of the Nazi Party.

 Following World War II he was imprisoned in Japan which transformed him spiritually.

 Upon returning to Germany he became a leading proponent of the Western esoteric spiritual tradition, synthesizing teachings from Christian Mysticism, Depth
Psychology and Zen Buddhism.
</item 5>

<item 6>
Katherine Dienes
Katherine Dienes (born 1970) is a New Zealand-born organist, conductor and composer.

 She is currently Organist and Master of the Choristers at Guildford Cathedral and is the first woman to hold the most senior musical post in a Church of
England cathedral.

 Her husband is Patrick Williams - librarian of the Royal Phillharmonic Orchestra, and they have a daughter, Hannah, who is currently a chorister at Guildford
Cathedral
</item 6>

<item 7>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 7>

<item 8>
David Draiman
David Michael Draiman (born March 13, 1973) is an American songwriter and the vocalist for the band Disturbed as well as for the band Device.

 Draiman is known for his distorted voice and percussive singing style.

 In November 2006, Draiman was voted number 42 on the "Hit Parader"’s ""Top 100 Metal Vocalists of All Time"".

 Draiman has written some of Disturbed's most successful singles, such as "Stupify", "Down with the Sickness", "Indestructible", and "Inside the Fire".
</item 8>

<item 9>
Device (metal band)
Device was an industrial metal band started by David Draiman, frontman of the heavy metal group Disturbed.

 Draiman was approached by Geno Lenardo, former guitarist of Filter, and together they started to work on new material.

 The two started work on a debut album in June 2012.

 The result of those sessions, "Device", was released on April 9, 2013.

 The first single and third track, "Vilify" was released to radio ahead of the album on February 19, 2013, alongside its first music video, directed by P. R.
Brown.

 The second single and the introductory track, "You Think You Know", was released on June 11, 2013, alongside its music video, directed once again by Brown.
</item 9>

<item 10>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 10>
=== END ITEM id=e35cdf2251637644 ===

=== ITEM id=0599ab07298a90a1 ===
Gold sentences and facts:
Sentence 0:  "Pinus" is the sole genus in the subfamily Pinoideae.
  Fact 0.0: Pinus is the sole genus in the subfamily Pinoideae.
Sentence 1:  The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35 unresolved species and many more synonyms.
  Fact 1.0: The Plant List was compiled by the Royal Botanic Gardens, Kew and the Missouri Botanical Garden.
  Fact 1.1: The Plant List accepts 126 species names of pines as current.
  Fact 1.2: The Plant List includes 35 unresolved species of pines.
  Fact 1.3: The Plant List includes many more synonyms.
Sentence 2:  It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
  Fact 2.0: The genus is sometimes considered to have only two species, B. monosperma and B. superba.
  Fact 2.1: The genus is sometimes expanded to include four or five species.

Retrieved items:
<item 1>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 1>

<item 2>
An encyclopedia-style collection spans plant biology, ecology, wildlife, food, and culture. It covers herbicides (imazapyr and HPPD inhibitors), plant defenses
and diversity, Huernia’s fly-attracting flowers, Oenanthe javanica, and Nepenthes pitcher plants, especially in Southeast Asia. Wildlife entries include
butterflies, birds, ticks, and deer, from large Sri Lankan sambar to threatened Asian species. Other topics include global and regional restaurants, hot sauces
</item 2>

<item 3>
The text covers several flowering plants. Pterostyrax is a four-species East Asian genus of deciduous shrubs and small trees; P. hispidus grows to 15 m and
bears fragrant white summer flowers. Styrax comprises about 130 shrubs and trees, mostly from Asia; its resin, benzoin, is used in perfume, incense, flavoring,
and medicine. Listed Styrax species are native to Malaysia, Singapore, Vietnam, and Peru;
</item 3>

<item 4>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 4>

<item 5>
An encyclopedia-style collection covers deer and other wildlife, noting habitats, ranges, traits, and conservation status: Sri Lankan sambar are exceptionally
large; moose are solitary northern browsers; several Asian deer are threatened, while Indian muntjac is least concern. It also mentions the Calypso
fish-identification database, wild boar, and black bull. Other entries describe the Umba River and Babati; Ovambo, Oku, and Ewe peoples; Hawaiian muumuus
</item 5>

<item 6>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 6>

<item 7>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 7>

<item 8>
Nepenthes is a genus of roughly 150 Old World tropical pitcher-plant species, richest in Borneo, Sumatra, and the Philippines; monkeys sometimes drink rainwater
from its pitchers. Bornean N. boschiana (with N. borneensis a synonym) is allied to N. faizaliana; N. chaniana has long white hairs and pale cylindrical
pitchers. N. × kinabaluensis, a cross of N. rajah and N. vil
</item 8>

<item 9>
Pine
A pine is any conifer in the genus Pinus, , of the family Pinaceae.

 "Pinus" is the sole genus in the subfamily Pinoideae.

 The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35
unresolved species and many more synonyms.
</item 9>

<item 10>
A botanical survey covers African Huernia, whose carrion-scented flowers attract flies; Asian water vegetable Oenanthe javanica; and many plant genera,
families, and species, with California botanist Samuel Parish commemorated in plant names. Nepenthes comprises about 150 Old World tropical pitcher plants,
especially diverse in Borneo, Sumatra, and the Philippines; monkeys drink pitcher rainwater, and several Bornean species and hybrids are noted. Other highlights
include a
</item 10>
=== END ITEM id=0599ab07298a90a1 ===
