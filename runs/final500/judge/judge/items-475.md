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

This file has 11 items.

=== ITEM id=52053ee7ccaea1ba ===
Gold sentences and facts:
Sentence 0: 3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.
  Fact 0.0: 3:10 to Yuma is a 2007 American Western film.
  Fact 0.1: 3:10 to Yuma was directed by James Mangold.
  Fact 0.2: 3:10 to Yuma was produced by Cathy Konrad.
  Fact 0.3: Russell Crowe and Christian Bale star in the lead roles of 3:10 to Yuma.
  Fact 0.4: Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman have supporting performances in 3:10 to Yuma.
Sentence 1: Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and Wat in "A Knight's Tale".
  Fact 1.0: Alan Wray Tudyk was born on March 16, 1971.
  Fact 1.1: Alan Tudyk is an American actor.
  Fact 1.2: Alan Tudyk is an American voice actor.
  Fact 1.3: Alan Tudyk is known for playing Hoban "Wash" Washburne in the space western television series "Firefly" and in the movie "Serenity".
  Fact 1.4: Alan Tudyk is known for playing Alpha in the science fiction TV series "Dollhouse".
  Fact 1.5: Alan Tudyk is known for playing Tucker McGee in "Tucker & Dale vs. Evil".
  Fact 1.6: Alan Tudyk is known for playing Steve the Pirate in "" (the title is missing from the source sentence).
  Fact 1.7: Alan Tudyk is known for playing Wat in "A Knight's Tale".

Retrieved items:
<item 1>
3:10 to Yuma (2007 film)
3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead
roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.

 It is about a drought-impoverished rancher (Bale) who takes on the dangerous job of taking a notorious outlaw (Crowe) to justice.

 It is a remake of the , making it the second adaptation of Elmore Leonard's short story "Three-Ten to Yuma".

 Filming took place in various locations in New Mexico.

 "3:10 to Yuma" opened September 7, 2007, in the United States and received positive reviews from critics.
</item 1>

<item 2>
The Appaloosa
The Appaloosa (also known as Southwest to Sonora) is a 1966 American Western film Technicolor (set in the 1870s) from Universal Pictures starring Marlon Brando,
Anjanette Comer and John Saxon, who was nominated for a Golden Globe for Best Supporting Actor for his portrayal of a Mexican bandit.

 The film was directed by Sidney J. Furie, and shot in Wrightwood, Antelope Valley, and Lake Los Angeles, California, St. George, Utah, and Colorado City,
Arizona.
</item 2>

<item 3>
Henry Murdoch
Henry Murdoch (17 September, 1920 - 24 April, 1987), born as George Henry Murdock, was an Australian aboriginal actor and stockman who appeared in Australian
films of the 1940s and 1950s.

 He was working as stockman in Rockhampton when discovered by Ralph Smart, who was helping make "The Overlanders" (1946).

 The film's director, Harry Watt, later claimed Murdoch and fellow aboriginal actor Clyde Combo "proved to be first-class actors and were exceedingly quick
witted and intelligent.

 They certainly disproved the conventional idea that the Australian aboriginal is an animalistic caveman."
</item 3>

<item 4>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 4>

<item 5>
Alan Tudyk
Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television
series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and
Wat in "A Knight's Tale".

 He also co-starred on the ABC sitcom "Suburgatory".

 Tudyk had a supporting role in the film "42" as baseball manager Ben Chapman and voiced roles in every Walt Disney Animation Studios film to date starting with
"Wreck-It Ralph".

 He has also featured in the web series "Con Man" and the films "" and "Rogue One: A Star Wars Story".
</item 5>

<item 6>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 6>

<item 7>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 7>

<item 8>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 8>

<item 9>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 9>

<item 10>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 10>
=== END ITEM id=52053ee7ccaea1ba ===

=== ITEM id=293f5dfff71377ef ===
Gold sentences and facts:
Sentence 0: Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on the German island of Greifswalder Oie in December 1937.
  Fact 0.0: Operation Lighthouse was the name given to a failed experimental launch.
  Fact 0.1: The Operation Lighthouse launch involved four Aggregate 3 liquid-fuel rockets.
  Fact 0.2: Wernher von Braun conducted the Operation Lighthouse launch.
  Fact 0.3: Walter Dornberger conducted the Operation Lighthouse launch.
  Fact 0.4: The Operation Lighthouse launch took place on the German island of Greifswalder Oie.
  Fact 0.5: The Operation Lighthouse launch took place in December 1937.
Sentence 1: Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War II.
  Fact 1.0: Major-General Dr. Walter Robert Dornberger was born on 6 September 1895.
  Fact 1.1: Walter Robert Dornberger died on 27 June 1980.
  Fact 1.2: Walter Robert Dornberger was a German Army artillery officer.
  Fact 1.3: Walter Robert Dornberger's career spanned World War I.
  Fact 1.4: Walter Robert Dornberger's career spanned World War II.

Retrieved items:
<item 1>
Magnus von Braun
Magnus "Mac" Freiherr von Braun (10 May 1919 – 21 June 2003) was a German chemical engineer, Luftwaffe aviator, and rocket scientist at Peenemünde, the
Mittelwerk, and after emigrating to the United States via Operation Paperclip, at Fort Bliss.

 He was the brother of Sigismund and Wernher von Braun.
</item 1>

<item 2>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 2>

<item 3>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
Man Will Conquer Space Soon!
Man Will Conquer Space Soon!

 was the title of a famous series of 1950s magazine articles in "Collier's" detailing Wernher von Braun's plans for manned spaceflight.

 Edited by Cornelius Ryan, the individual articles were authored by such space notables of the time as Willy Ley, Fred Lawrence Whipple, Dr. Joseph Kaplan, Dr.
Heinz Haber, and von Braun.

 The articles were illustrated with paintings and drawings by Chesley Bonestell, Fred Freeman, and Rolf Klep, some of the finest magazine illustrators of the
time.
</item 5>

<item 6>
Von Braun (crater)
von Braun, named after the rocket pioneer Wernher von Braun, is a lunar crater located near the northwestern limb of the Moon.

 It lies along the western edge of the Oceanus Procellarum, to the northeast of the crater Lavoisier.

 The northeastern rim of this crater is on the edge of the Sinus Roris, a bay feature in the northwestern part of the Oceanus Procellarum.

 Due to its proximity to the limb, this crater appears significantly foreshortened when viewed from the Earth.
</item 6>

<item 7>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 7>

<item 8>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 8>

<item 9>
Von Braun Ferry Rocket
Von Braun Ferry Rocket was a concept design for a shuttle spacecraft that was developed by Wernher von Braun in a seminal series of early-1950s Collier's
magazine articles, Man Will Conquer Space Soon!

 by Wernher von Braun "et al." The Ferry Rocket concept has evolved over time.
</item 9>

<item 10>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 10>
=== END ITEM id=293f5dfff71377ef ===

=== ITEM id=30058e1060adb650 ===
Gold sentences and facts:
Sentence 0: Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca O'Brien and Lauren Dark.
  Fact 0.0: Mediastan is a documentary film.
  Fact 0.1: Mediastan was released in 2013.
  Fact 0.2: Mediastan is about the 2010 United States diplomatic cables leak.
  Fact 0.3: Mediastan was directed by Johannes Wahlström.
  Fact 0.4: Mediastan was produced by Julian Assange.
  Fact 0.5: Mediastan was produced by Rebecca O'Brien.
  Fact 0.6: Mediastan was produced by Lauren Dark.
Sentence 1: Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he founded in 2006.
  Fact 1.0: Julian Paul Assange was born Julian Paul Hawkins.
  Fact 1.1: Julian Paul Assange was born on 3 July 1971.
  Fact 1.2: Julian Paul Assange is Australian.
  Fact 1.3: Julian Paul Assange is a computer programmer.
  Fact 1.4: Julian Paul Assange is the founder of WikiLeaks.
  Fact 1.5: WikiLeaks is an organisation.
  Fact 1.6: Julian Paul Assange founded WikiLeaks in 2006.

Retrieved items:
<item 1>
Julian Assange
Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he
founded in 2006.

 He has won accolades including the Sam Adams Award and Martha Gellhorn Prize for Journalism.
</item 1>

<item 2>
Mediastan
Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca
O'Brien and Lauren Dark.
</item 2>

<item 3>
Amal Clooney
Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and
human rights.

 Her clients include Julian Assange, the founder of WikiLeaks, in his fight against extradition.

 She has also represented the former prime minister of Ukraine, Yulia Tymoshenko, and Egyptian-Canadian journalist Mohamed Fahmy.

 She is married to the American actor George Clooney.
</item 3>

<item 4>
Podsafe Music Network
Podsafe Music Network (PMN) was the primary archive of podsafe music (music available for use in podcasting without significant licensing difficulties) on the
internet.

 It was established by Mevio, a podcast production company founded by Adam Curry and Ron Bloom, in the summer of 2005.

 The network's website provides tools for musicians and for podcasters, and also makes streamed music available to the casual listener.
</item 4>

<item 5>
Midnite Movies
Midnite Movies is a line of B movies released first on VHS and later on DVD by MGM Home Entertainment.

 The line was begun by MGM in March 2001 following its acquisition of Orion Pictures, which bought out Filmways, the owner of American International Pictures.

 AIP had a library of B movies from the 1950s and 1960s that were science fiction, horror, and exploitation films.

 The "Midnite Movies" collection is primarily derived from the AIP library (including most of Roger Corman's and Vincent Price's horror movies) but also
included Hammer Film Productions, Amicus Productions, United Artists, and Empire International Pictures movies as well.

 The DVDs were first released as single films but most later releases would be double features on single double-sided discs.

 Sony Pictures Home Entertainment later became owners of the MGM library and continued the "Midnite Movies" line with distributor 20th Century Fox Home
Entertainment.

 All double feature titles released on the 20th Century Fox label were two-disc packages.

 By 2011, no new titles were forthcoming; the previous catalog titles slowly went out of print and the "Midnite Movies" website was taken down.
</item 5>

<item 6>
Music West Records
Music West Records was an independent record company founded by Allan Kaplan on December 1985 in San Rafael, California.

 The company was initially formed to promote Ray Lynch.

 During its run, artists released under the record company included Jim Chappell, Kenneth Nash, Chris Spheeris, and Øystein Sevåg.

 According to Gary Chappell, the manufacturer for Music West, the artists originated independently, claiming that the company's idea "has a statement that comes
directly from the
</item 6>

<item 7>
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 7>

<item 8>
Shamal Sabri
Shamal Sabri (Kurdish: شه‌مال سه‌بری), (born 9 November 1985), is an independent Kurdish award-winning film producer and Duhok International Film Festival
Artistic Director, graduated from Duhok University with BA in English Literature.

.

 He was born in the city of Duhok (Kurdish: دهوك) in Iraqi Kurdistan.

 Apart from being one of the co-founders of Duhok IFF in 2009, he has worked in many other fields of video productions, produced many commercials and music
videos in NV Production and Vin TV.

 Throughout his film carrier, he has contributed in many award-winning films as an assistant director, production manager, assistant producer and line producer.

 His latest contributions were in many films such as "Before Snowfall" by Hisham Zaman, "The Swallow" by Mano Khalil, "Memories on Stone" by Shawkat Amin Korki
and Michael by Kurdo Duski.
</item 8>

<item 9>
Micom Film and TV Productions
It was a family based company with long tradition in show and entertainment business.

 Micom produced around 30.000 minutes of program for National TV Slovenia and other media.

 In 2002 company stopped producing because of the tragedy in the family.

 Both sons went separate way and build their own Film and TV production house.
</item 9>

<item 10>
Micom Film and TV Productions
MICOM Film and TV Productions was established in 1989 by Sreco Mihelcic in a Former Yugoslavian republic Slovenia.

 In the early 90' company joined two sons, Matjaz Mihelcic - Program director, Producer and Vasja Mihelcic - technical director and Executive Producer.

 MICOM was in 1990s the leading Slovenian creative house, producing TV shows for youth, popular entertainment TV shows and discovered many new TV concepts.

 Micom discovered many talents, different creative people and gave them possibility to express themselves.

 The most popular TV shows were: Lahkih nog naokrog (educational youth TV series), Cari zacimb (entertaining cooking TV series), Razpoke v casu (fantasy TV
series).

 They also produced documentary movies.

 One of them got in 1996 internationally release - Fight for the river and it's distributed by International Historic Films.

.

 In the beginning of 2000 company joined Marjana Mihelcic.

 It was a family based company with long tradition in show and entertainment business.
</item 10>
=== END ITEM id=30058e1060adb650 ===

=== ITEM id=7a4a53c833c8d68b ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province, People's Republic of China.
Sentence 1: Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.
  Fact 1.0: Hailar District's Mongolian name is ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ, romanized as Qayilar toɣoriɣ.
  Fact 1.1: Hailar District's Mongolian name in Cyrillic script is Хайлаар тойрог.
  Fact 1.2: Hailar District's Cyrillic name is Хайлаар дүүрэг, romanized as Khailaar düüreg.
  Fact 1.3: Hailar District is an urban district.
  Fact 1.4: Hailar District serves as the seat of the prefecture-level city Hulunbuir.
  Fact 1.5: Hailar District is located in northeastern Inner Mongolia, China.

Retrieved items:
<item 1>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 1>

<item 2>
Hailar District
Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as
the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.

 Hulunbuir, due to its massive size, is a city in administrative terms only, being mainly grassland and rural.

 Hailar, therefore, is a "de facto" city.

 Hailar can also refer to the urban area around the Hailar district, with Hulunbuir being the wider geographical region that contains the urban area.
</item 2>

<item 3>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 3>

<item 4>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 4>

<item 5>
Old Barag Banner
Old Barag Banner (Mongolian: ᠬᠠᠭᠤᠴᠢᠨ ᠪᠠᠷᠭᠤ ᠬᠣᠰᠢᠭᠤ Қаучин Бару қосиу "Qaɣučin Barɣu qosiɣu"; ) is a banner of northeastern Inner Mongolia, People's Republic of
China.

 It is located 28 km from Hailar District, the administrative centre of Hulunbuir City, which administers this banner.
</item 5>

<item 6>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 6>

<item 7>
Hulunbuir
Hulunbuir or Hulun Buir (Mongolian: ; , "Hūlúnbèi'ěr") is a region that is governed as a prefecture-level city in northeastern Inner Mongolia, in China.

 Its administrative center is located at Hailar District, its largest urban area.

 Major scenic features are the high steppes of the Hulun Buir grasslands, the Hulun and Buir lakes (the latter partially in Mongolia), and the Khingan range.

 Hulun Buir borders Russia to the north and west, Mongolia to the south and west, Heilongjiang province to the east and Hinggan League to the direct south.

 Hulunbuir is a linguistically diverse area: next to Mandarin Chinese, Mongolian dialects such as Khorchin and Buryat, the Mongolic language Dagur and some
Tungusic languages are spoken there.
</item 7>

<item 8>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 8>

<item 9>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 9>

<item 10>
Ludhiana
Ludhiana is a city and a municipal corporation in Ludhiana district in the Indian state of Punjab, and is the largest city north of Delhi.

 It is the largest city in the state, with an area of 310sq.

 km and an estimated population of 1,789,650 as of the 2015.

 The population increases substantially during the harvesting season due to the migration of labourers from the eastern states of Uttar Pradesh, Bihar, Odisha.

 The city stands on the Sutlej River's old bank, 13 km south of its present course.

 It is an industrial centre of northern India, and was referred to as India's Manchester by the BBC.
</item 10>
=== END ITEM id=7a4a53c833c8d68b ===

=== ITEM id=e973bbdffbf214a7 ===
Gold sentences and facts:
Sentence 0: Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.
  Fact 0.0: Edward Benjamin Britten, Baron Britten, lived from 22 November 1913 to 4 December 1976.
  Fact 0.1: Edward Benjamin Britten, Baron Britten, was an English composer.
  Fact 0.2: Edward Benjamin Britten, Baron Britten, was an English conductor.
  Fact 0.3: Edward Benjamin Britten, Baron Britten, was an English pianist.
Sentence 1: Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.
  Fact 1.0: Béla Viktor János Bartók lived from 25 March 1881 to 26 September 1945.
  Fact 1.1: Béla Viktor János Bartók was a Hungarian composer.
  Fact 1.2: Béla Viktor János Bartók was a Hungarian pianist.
  Fact 1.3: Béla Viktor János Bartók was a Hungarian ethnomusicologist.

Retrieved items:
<item 1>
Benjamin Britten
Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.

 He was a central figure of 20th-century British classical music, with a range of works including opera, other vocal music, orchestral and chamber pieces.

 His best-known works include the opera "Peter Grimes" (1945), the "War Requiem" (1962) and the orchestral showpiece "The Young Person's Guide to the Orchestra"
(1945).
</item 1>

<item 2>
Graph query row:
person: Béla Bartók
nationality: Hungarian
country_of_origin: Hungarian
associated_countries: ['Saanen', 'Budapest', 'Hungary']
</item 2>

<item 3>
List of string quartets by Béla Bartók
1 and 28 both owe a great deal to Bartók's quartets (; ), György Ligeti, whose two string quartets both owe a great deal to Bartók’s quartets (; ), Bruno
Maderna , George Perle, who credits the Bartók Fourth and Fifth Quartets as precedents for his use of arrays of chords related to one another by different types
of symmetry , Walter Piston (; ), Kim Dzmitrïyevich Tsesakow , Wilfried Westerlinck , Stefan Wolpe, who explained in a public lecture how he had derived ideas
from Bartók’s Fourth Quartet , and Xu Yongsan .
</item 3>

<item 4>
Graph query row:
person: Benjamin Britten
nationality: English
country_of_origin: English
associated_countries: []
</item 4>

<item 5>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 5>

<item 6>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 6>

<item 7>
String Quartet No. 6 (Bartók)
The String Quartet No. 6, Sz.

 114, BB 119, was the final string quartet that Béla Bartók wrote before his death.

 It was begun in August 1939 in Saanen, Switzerland, where Bartók was a guest of his patron, the conductor Paul Sacher.

 Shortly after he completed the Divertimento for String Orchestra on the 17th, he started on a commission for his friend, the violinist Zoltán Székely.

 Székely was acting as intermediary for the "New Hungarian Quartet", who had given the Budapest premiere of the String Quartet No. 5.

 With the outbreak of World War II and his mother's illness, Bartók returned to Budapest, where the quartet was finished in November.

 After his mother's death, Bartók decided to leave with his family for the United States.
</item 7>

<item 8>
List of compositions by Béla Bartók
This aspires to be a complete list of compositions by Béla Bartók.

 The catalogue numbering by András Szőllősy (Sz.)

, László Somfai (BB.)

 and Denijs Dille (DD.)

 are provided, as well as Bartók's own opus numbers.

 Note that Bartók started three times anew with opus numbers, here indicated with "(list 1)", "(list 2)" and "(list 3)" respectively.

 The pieces from the third listing are by far best known; opus lists 1 and 2 are early works.

 The year of composition and instrumentation (including voice) are included.

 See the main article on Béla Bartók for more details.
</item 8>

<item 9>
Suite, Op. 14 (Bartók)
The Suite, Op. 14, Sz.

 62, BB 70 is a piece for solo piano written by Béla Bartók.

 It was written in February 1916, published in 1918, and debuted by the composer on April 21, 1919, in Budapest.

 The Suite is one of Bartók's most significant works for piano, only comparable with his 1926 Piano Sonata.

 Though much of Bartók's work makes frequent use of Eastern European folk music, this suite is one of the few pieces without melodies of folk origin.

 However, Romanian, Arabic, and North African rhythmic influences can still be found in some movements.

 Originally intending the suite to be a five-movement work, Bartók later decided against the idea and discarded the second movement, the "Andante", which was
published only posthumously in the October 1955 issue of "Új Zenei Szemle" (New Musical Review).
</item 9>

<item 10>
List of string quartets by Béla Bartók
The Hungarian composer Béla Bartók wrote six string quartets, each for the usual forces of two violins, viola and cello.

 Notable composers who have been influenced by them include Benjamin Britten, particularly in the Sonata in C for Cello and Piano (; ), Elliott Carter, who
refers in the opening of his own First String Quartet to Bartók’s Sixth Quartet , Chen Yi , Edison Denisov, whose Second Quartet is closely related to Bartók’s
Fifth Quartet , Franco Donatoni, who was deeply impressed when he heard a broadcast of Bartók's Fourth Quartet , Robert Fripp, who mentions them as an influence
upon King Crimson , Miloslav Ištvan , György Kurtág, whose Opp.
</item 10>
=== END ITEM id=e973bbdffbf214a7 ===

=== ITEM id=07e9700412bf7c5b ===
Gold sentences and facts:
Sentence 0:  It consists of three shots, two Sourz Apple and one Cherry Brandy.
  Fact 0.0: The described drink consists of three shots.
  Fact 0.1: The described drink has two shots of Sourz Apple.
  Fact 0.2: The described drink has one shot of Cherry Brandy.
Sentence 1:  Each shot is taken from an average shot glass which is around 1.5 ounces (roughly 45 ml).
  Fact 1.0: Each shot is taken from an average shot glass.
  Fact 1.1: An average shot glass holds around 1.5 ounces (roughly 45 ml).
Sentence 2: The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.
  Fact 2.0: The flaming volcano is a large tropical group cocktail.
  Fact 2.1: The flaming volcano is made of rum, brandy, pineapple juice, orange juice and almond syrup.
  Fact 2.2: The flaming volcano is sometimes made with other ingredients.
Sentence 3:  It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about 32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.
  Fact 3.0: The flaming volcano is usually a multi-user drink.
  Fact 3.1: The flaming volcano is served to a group in a special ritual vessel known as a volcano bowl.
  Fact 3.2: A volcano bowl is a decorative ceramic bowl.
  Fact 3.3: A volcano bowl typically has a capacity of about 32 USoz.
  Fact 3.4: A volcano bowl is designed with a rising central hub feature.
  Fact 3.5: The rising central hub feature of a volcano bowl usually resembles a crude volcano cone.

Retrieved items:
<item 1>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 1>

<item 2>
The material surveys wildlife and ecological traits, including butterflies, moths, birds, ticks, orchids, and Cretaceous herbivores; restaurant chains spanning
global fast food and regional cuisines; and hot sauces from Valentina and Tabasco to Puerto Rican pique. It also covers apple dumplings, mulling spices, Finnish
Christmas traditions, and cocktails such as Flaming Volcano and Blue Lagoon, noting the spectacle of flaming drinks.
</item 2>

<item 3>
Flaming Doctor Pepper
A Flaming Doctor Pepper is a flaming cocktail said to taste like the soft drink Dr Pepper, although Dr Pepper is not one of its ingredients.
</item 3>

<item 4>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 4>

<item 5>
Volcanic bomb
A volcanic bomb is a mass of molten rock (tephra) larger than 64 mm (2.5 inches) in diameter, formed when a volcano ejects viscous fragments of lava during an
eruption.

 They cool into solid fragments before they reach the ground.

 Because volcanic bombs cool after they leave the volcano, they are extrusive igneous rocks.

 Volcanic bombs can be thrown many kilometres from an erupting vent, and often acquire aerodynamic shapes during their flight.

 Bombs can be extremely large; the 1935 eruption of Mount Asama in Japan expelled bombs measuring 5–6 m in diameter up to 600 m from the vent.

 Volcanic bombs are a significant volcanic hazard, and can cause severe injuries and death to people in an eruption zone.

 One such incident occurred at Galeras volcano in Colombia in 1993; six people near the summit were killed and several seriously injured by lava bombs when the
volcano erupted unexpectedly.
</item 5>

<item 6>
Tweed Volcano
Tweed Volcano is a partially eroded Early Miocene shield volcano located in northeastern New South Wales, which formed when this region of Australia passed over
the East Australia hotspot around 23 million years ago.

 Mount Warning, Lamington Plateau and the Border Ranges between New South Wales and Queensland are among the remnants of this volcano that was originally over
100 km in diameter and nearly twice the height of Mount Warning today, at 1156 m .

 Despite its size, Tweed Volcano was not a supervolcano; other shield volcanoes - such as on Hawaii - are much larger.

 In the 23 million years since the volcano was active, erosion has been extensive, forming a large erosion caldera around the volcanic plug of Mount Warning.

 Its erosion caldera is the largest in the Southern Hemisphere.
</item 6>

<item 7>
Apple-Kneel
The Apple-Kneel is a cocktail.

 It consists of three shots, two Sourz Apple and one Cherry Brandy.

 Each shot is taken from an average shot glass which is around 1.5 ounces (roughly 45 ml).

 Firstly, you put two shots of Sourz Apple into a glass and then put in one shot of Cherry Brandy in and mix.

 It is customary to sip this cocktail.

 It can be served 'neat' or 'on the rocks'.
</item 7>

<item 8>
The entries cover foods, drinks, customs, and folklore. Apple dumplings are baked pastry-wrapped apples; mulling spices flavor warm seasonal drinks such as
mulled wine and Finnish glögi. Cocktails include Flaming Volcano (a rum-and-brandy tropical group drink served in an ignited volcano bowl), Flaming Doctor
Pepper, Apple-Kneel, and Blue Lagoon; flaming drinks use lit high-proof alcohol for spectacle or flavor. Other topics include Finland’s Christmas feast, the
fire
</item 8>

<item 9>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 9>

<item 10>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 10>
=== END ITEM id=07e9700412bf7c5b ===

=== ITEM id=a8baee06dc167050 ===
Gold sentences and facts:
Sentence 0:  It is an annual event hosted by the Department of Urology at the University of Virginia, and the Charlottesville Track Club.
  Fact 0.0: The event is an annual event.
  Fact 0.1: The event is hosted by the Department of Urology at the University of Virginia.
  Fact 0.2: The event is hosted by the Charlottesville Track Club.
Sentence 1: Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.
  Fact 1.0: Panera Bread Company is an American chain.
  Fact 1.1: Panera Bread Company operates bakery-café fast casual restaurants.
  Fact 1.2: Panera Bread Company's restaurants are in the United States.
  Fact 1.3: Panera Bread Company's restaurants are in Canada.

Retrieved items:
<item 1>
The European 10,000m Cup, organized annually by European Athletics since 1997, grew from the Iberian championships and was renamed in 2005. Track 10,000m races
cover 25 laps; the 1500m is a major Olympic middle-distance event. Charlottesville’s annual Bill Steers Men’s 4-Miler, held since 2004, promotes men’s health.
European under-20 championships evolved from junior games (1964, 1968)
</item 1>

<item 2>
Bill Steers Men's 4-Miler
The Bill Steers Men's 4-Miler (formerly Charlottesville Men's Four Miler) is the only all men's run in the Charlottesville area.

 The race was rededicated as a tribute to William D. Steers in 2015.

 It is an annual event hosted by the Department of Urology at the University of Virginia, and the Charlottesville Track Club.

 The event has also been sponsored by Panera Bread, BMW of Charlottesville, Starbucks, Pepsi and the City of Charlottesville.

 The City of Charlottesville also hosts the Women's Four Miler to benefit the UVA Cancer Center Breast Care Program.

 The event, held continuously since 2004, was conceived by William D. Steers with the purpose of raising awareness on men's health issues; to encourage men to
take control of their health, and become more physically active.
</item 2>

<item 3>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
</item 3>

<item 4>
Panera Bread is a U.S.–Canada bakery-café chain headquartered near St. Louis, where it is called Saint Louis Bread Company. Indian entrepreneur Prem Ganapathy
founded Dosa Plaza, growing it to 45 outlets across four countries. Quebec’s independently franchised La Belle Province is sometimes open 24 hours. Gino’s
Hamburgers began in Baltimore in 1957 and relaunched in 2010. West Virginia-based Tudor’s Biscuit World serves breakfast and homestyle
</item 4>

<item 5>
William D. Steers
Dr. William D. Steers, August 19, 1955 – April 10, 2015, was a Paul Mellon professor and Chair of the Department of Urology at the School of Medicine of the
University of Virginia.

 Dr. Steers is past President of the American Board of Urology (ABU) and Editor of the "Journal of Urology".

 In 2003, the University of Virginia awarded Dr. Steers the Hovey Dabney Professorship.

 In 2004 Dr. Steers initiated the Charlottesville Men’s Four Miler road race in Virginia to raise funds for men’s health.

 Dr. Steers was a viticulturist aficionado, he co-owned Well Hung Vineyard in Charlottesville.

 He also authored YOURometer an iPhone app.

 used to record urological related symptoms.
</item 5>

<item 6>
Team Hole in the Wall
Team Hole in the Wall is an athletic fundraising initiative managed by the Hole in the Wall Gang Camp.

 Amateur athletes join up to raise a specified amount of funds and receive entrance to a marathon, bike ride or other athletic event in support of seriously ill
children served by The Hole in the Wall Gang Camp or one of its associated camps around the world.

 Established in 2005, Team Hole in the Wall offers entrance into more than two dozen marathons and cycling events, including the New York City Marathon, Boston
Marathon and AngelRide cycling event in Connecticut.

 New Canaan resident and Newman's Own Vice President of Marketing Michael “Mike” Havard ran in the ING New York City Marathon prior to 2004, but in that year he
decided to use his participation to raise funds for The Hole in the Wall Gang Camp, raising nearly $40,000.

 The next year, he served as captain of the first Team Hole in the Wall team of 40 runners, raising $150,000.

 The number of Team Hole in the Wall members and events has risen each year since.

 In 2010, more than 1,800 Team Hole in the Wall members are expected to participate in 20 athletic events.
</item 6>

<item 7>
Terry Baker
Terry Wayne Baker (born May 5, 1941) is a former American football and basketball player.

 He played college football and college basketball at the Oregon State University.

 He played as a quarterback for the Oregon State Beavers football team from 1960 to 1962, winning the Heisman Trophy as senior.

 In the spring of his senior year, he played in the Final Four of the 1963 NCAA Men's Division I Basketball Tournament with the Oregon State Beavers men's
basketball team.

 To date, he is the only athlete to win a Heisman Trophy and play in the Final Four.

 Baker was the first overall pick in the 1963 NFL draft and played with the Los Angeles Rams of the National Football League (NFL) from 1963 to 1965.

 He then played for one season in the Canadian Football League (CFL) with the Edmonton Eskimos, in 1966.

 Baker was inducted into the College Football Hall of Fame in 1982.
</item 7>

<item 8>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
</item 8>

<item 9>
Tudor's Biscuit World
Tudor's Biscuit World is a restaurant chain based in Huntington, West Virginia, most commonly found in West Virginia.

 Many West Virginia locations share a building with Gino's Pizza and Spaghetti, although the chain is more extensive than Gino's (which is exclusive to West
Virginia), having locations in southern Ohio, eastern Kentucky, and southwestern Virginia.

 In 2016 a franchise was opened in Panama City, Florida.

 Tudor's serves biscuits, biscuit sandwiches, homestyle breakfasts and dinners, muffins, and several side dishes.

 The chain was originally based in Charleston, West Virginia and many of the biscuit sandwiches are named for sports teams of interest in that area, including
teams at Marshall University, West Virginia University, and The University of Charleston.
</item 9>

<item 10>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 10>
=== END ITEM id=a8baee06dc167050 ===

=== ITEM id=f36f532e274e90ae ===
Gold sentences and facts:
Sentence 0: Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.
  Fact 0.0: Alice Ann Munro's nee name is Laidlaw.
  Fact 0.1: Alice Ann Munro was born on 10 July 1931.
  Fact 0.2: Alice Ann Munro is a Canadian short story writer.
  Fact 0.3: Alice Ann Munro is a Nobel Prize winner.
Sentence 1: Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well as her left-wing sympathies and political activism.
  Fact 1.0: Lillian Florence Hellman was born on June 20, 1905.
  Fact 1.1: Lillian Florence Hellman died on June 30, 1984.
  Fact 1.2: Lillian Florence Hellman was an American dramatist.
  Fact 1.3: Lillian Florence Hellman was an American screenwriter.
  Fact 1.4: Lillian Florence Hellman is known for her success as a playwright on Broadway.
  Fact 1.5: Lillian Florence Hellman is known for her left-wing sympathies.
  Fact 1.6: Lillian Florence Hellman is known for her political activism.

Retrieved items:
<item 1>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 1>

<item 2>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 2>

<item 3>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 3>

<item 4>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 4>

<item 5>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 5>

<item 6>
The entries profile writers, scholars, artists, publishers, and advocates. Highlights include computer scientist Nuria Oliver’s data-science leadership and
human-behavior research; Helen Keller’s childhood story “The Frost King”; Booker Prize winner Keri Hulme; Alice Walker’s Pulitzer-winning *The Color Purple* and
anti-FGM work; and Lucille Watahomigie’s Hualapai language revitalization. Others include novelists Sarra Manning and Kim Edwards, poet Tara Betts
</item 6>

<item 7>
The entries profile women in literature, theatre, music, visual art, business, and activism. Highlights include Pearson CEO Marjorie Scardino, the first woman
to lead a FTSE 100 company; Nobel-winning short-story writer Alice Munro; actress Elizabeth Taylor; pioneering Restoration writer Aphra Behn; and Aboriginal
activist Mum Shirl. They also cover Lucy Maud Montgomery’s Anne novels, Jean Webster’s *When Patty Went to College*, Margaret Landon’s *Anna and the King
</item 7>

<item 8>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 8>

<item 9>
The Little Foxes (film)
The Little Foxes (1941) is an American drama film directed by William Wyler.

 The screenplay by Lillian Hellman is based on her 1939 play of the same name.

 Hellman's ex-husband Arthur Kober, Dorothy Parker and her husband Alan Campbell contributed additional scenes and dialogue.
</item 9>

<item 10>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 10>
=== END ITEM id=f36f532e274e90ae ===

=== ITEM id=505895caa500696c ===
Gold sentences and facts:
Sentence 0: Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and medical teaching facilities in Saint John, New Brunswick.
  Fact 0.0: Dalhousie University, commonly known as Dal, is a public research university.
  Fact 0.1: Dalhousie University is in Nova Scotia, Canada.
  Fact 0.2: Dalhousie University has three campuses in Halifax.
  Fact 0.3: Dalhousie University has a fourth campus in Bible Hill.
  Fact 0.4: Dalhousie University has medical teaching facilities in Saint John, New Brunswick.
Sentence 1: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 1.0: California State Polytechnic University, Pomona is also known as CPP, Cal Poly Pomona, or Cal Poly.
  Fact 1.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 1.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.

Retrieved items:
<item 1>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 1>

<item 2>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 2>

<item 3>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 3>

<item 4>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 4>

<item 5>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 5>

<item 6>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 6>

<item 7>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 7>

<item 8>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 8>

<item 9>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 9>

<item 10>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 10>
=== END ITEM id=505895caa500696c ===

=== ITEM id=a2f8a91deec97761 ===
Gold sentences and facts:
Sentence 0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.
  Fact 0.0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held.
  Fact 0.1: The 2015 AMVCA was held on March 7, 2015.
  Fact 0.2: The 2015 AMVCA was held at the Eko Hotel and Suites.
  Fact 0.3: The Eko Hotel and Suites is in Victoria Island, Lagos, Nigeria.
Sentence 1: Lagos initially emerged as a port city which originated on a collection of islands, which are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa; the islands are separated by creeks, fringing the southwest mouth of Lagos Lagoon, while protected from the Atlantic Ocean by barrier islands and long sand spits such as Bar Beach, which stretch up to 100 km east and west of the mouth.
  Fact 1.0: Lagos initially emerged as a port city.
  Fact 1.1: Lagos originated on a collection of islands.
  Fact 1.2: The islands are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa.
  Fact 1.3: The islands are separated by creeks.
  Fact 1.4: The creeks fringe the southwest mouth of Lagos Lagoon.
  Fact 1.5: The islands are protected from the Atlantic Ocean by barrier islands.
  Fact 1.6: The islands are protected from the Atlantic Ocean by long sand spits such as Bar Beach.
  Fact 1.7: The barrier islands and long sand spits, such as Bar Beach, stretch up to 100 km east and west of the mouth.

Retrieved items:
<item 1>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 1>

<item 2>
Africa Magic
Africa Magic is a collection of Pay TV entertainment channels that focus on African programming.

 "Africa Magic", which started off as single channel of the same name, is a brand owned by M-Net and now comprises eight channels.

 The first "Africa Magic" channel was launched in July 2003 as a movie channel and over the next decade, the brand expanded to include seven more channels
comprising movies, television shows and general entertainment.

 Africa Magic currently broadcasts in more than 50 African countries.

 The channels include "Africa Magic Family", "Africa Magic World", "Africa Magic Showcase", "Africa Magic Yoruba", "Africa Magic Igbo", "Africa Magic Hausa".

 "Africa Magic Epic" and "Africa Magic Urban".

 Africa Magic is also responsible for the annual Africa Magic Viewers' Choice Awards (AMVCAs), the biggest celebration of film and television talent in Africa.
</item 2>

<item 3>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 3>

<item 4>
2015 Africa Magic Viewers Choice Awards
The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.

 IK and Vimbai were the hosts of the event.

 A new category, Best Indigenous Language (Igbo), was introduced in the 2015 awards.
</item 4>

<item 5>
Eric Aghimien
Eric Enomamien Aghimien is a Nigerian director, producer, screenwriter and editor.

 His debut feature film, "A Mile from Home" won awards at both the 2014 Africa Magic Viewers Choice Awards and the 10th Africa Movie Academy Awards.
</item 5>

<item 6>
2013 Africa Magic Viewers Choice Awards
The 2013 Africa Magic Viewers Choice Awards were held on 9 March 2014 and co-hosted by Big Brother Africa presenter IK Osakioduwa and StarGist’s host, Vimbai
Mutinhiri.

 Ivie Okujaiye and Olu Jacobs were honored with the TrailBlazer and Industry Merit Awards respectively.
</item 6>

<item 7>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 7>

<item 8>
Lagos
Lagos initially emerged as a port city which originated on a collection of islands, which are contained in the present day Local Government Areas (LGAs) of
Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa; the islands are separated by creeks, fringing the southwest mouth of Lagos Lagoon, while protected from the
Atlantic Ocean by barrier islands and long sand spits such as Bar Beach, which stretch up to 100 km east and west of the mouth.

 Due to rapid urbanization, the city expanded to the west of the lagoon to include areas in the present day Lagos Mainland, Ajeromi-Ifelodun and Surulere.

 This led to the classification of Lagos into two main areas - the Island, which was the initial city of Lagos, before it expanded into the area known as the
Mainland.

 This city area was governed directly by the Federal Government through the "Lagos City Council", until the creation of Lagos State in 1967, which led to the
splitting of Lagos city into the present day seven Local Government Areas (LGAs), and an addition of other towns (which now make up 13 LGAs) from the then
Western Region, to form the state.
</item 8>

<item 9>
Lake Worth Lagoon
The Lake Worth Lagoon is a lagoon located in Palm Beach County, Florida.

 It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.

 The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
</item 9>

<item 10>
Africa Magic Viewers' Choice Awards
Africa Magic Viewers' Choice Awards (AMVCA) is an annual accolade presented by Multichoice recognizing outstanding achievement in television and film, voted on
by the general public.

 The inaugural Africa Magic Viewers’ Choice Awards ceremony was held in Lagos, Lagos State in Nigeria on 9 March 2013, and was broadcast live in more than 50
countries.

 Entries into the award ceremony are films and TV series that have been aired on Dstv Channels in the previous year.
</item 10>
=== END ITEM id=a2f8a91deec97761 ===

=== ITEM id=f621e08b9ac6db9d ===
Gold sentences and facts:
Sentence 0: Inner Secrets is the ninth studio album by Santana.
  Fact 0.0: Inner Secrets is a studio album by Santana.
  Fact 0.1: Inner Secrets is Santana's ninth studio album.
Sentence 1: Carlos Santana    (born July 20, 1947) is a Mexican and American musician who first became famous in the late 1960s and early 1970s with his band, Santana, which pioneered a fusion of rock and Latin American music.
  Fact 1.0: Carlos Santana was born on July 20, 1947.
  Fact 1.1: Carlos Santana is a Mexican musician.
  Fact 1.2: Carlos Santana is an American musician.
  Fact 1.3: Carlos Santana first became famous in the late 1960s and early 1970s.
  Fact 1.4: Carlos Santana first became famous with his band Santana.
  Fact 1.5: The band Santana pioneered a fusion of rock and Latin American music.

Retrieved items:
<item 1>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 1>

<item 2>
Inner Secrets
Inner Secrets is the ninth studio album by Santana.

 It was released in 1978 and marks the start of the phase of Santana's career where he moved away from the fusion of Latin, jazz, rock and blues that marked his
previous records and began to move towards an album-oriented rock direction.

 As such, the album's quality is widely disputed among fans. "

Stormy" and "One Chain (Don't Make No Prison)" were both hit singles.

 In The Netherlands "Well All Right" was released as a single and reached #22 in the top 40.
</item 2>

<item 3>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 3>

<item 4>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 4>

<item 5>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 5>

<item 6>
Carlos Santana
Carlos Santana    (born July 20, 1947) is a Mexican and American musician who first became famous in the late 1960s and early 1970s with his band, Santana,
which pioneered a fusion of rock and Latin American music.

 The band's sound featured his melodic, blues-based guitar lines set against Latin and African rhythms featuring percussion instruments such as timbales and
congas not generally heard in rock music.

 Santana continued to work in these forms over the following decades.

 He experienced a resurgence of popularity and critical acclaim in the late 1990s.

 In 2003 "Rolling Stone" magazine listed Santana at number 20 on their list of the 100 Greatest Guitarists of All Time.

 He has won 10 Grammy Awards and three Latin Grammy Awards.
</item 6>

<item 7>
Ex Pistols
The Ex Pistols were a scam act introduced in 1979 by one-time Sex Pistols producer Dave Goodman after his services were substituted for those of professional
producers.

 The Ex Pistols existed as a sound-alike band meant to mislead fans due to the similar music, name, and artwork, that led people to believe their songs were
actually new or lost Sex Pistols recordings.
</item 7>

<item 8>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 8>

<item 9>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 9>

<item 10>
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 10>
=== END ITEM id=f621e08b9ac6db9d ===
