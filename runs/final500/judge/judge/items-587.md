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

=== ITEM id=6aa6230604951d6e ===
Gold sentences and facts:
Sentence 0: In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
  Fact 0.0: The Fader published an interview with CupcakKe in September.
  Fact 0.1: In the interview, CupcakKe discussed her new album.
  Fact 0.2: CupcakKe's new album would address "rape, and everything else".
Sentence 1: The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.
  Fact 1.0: The Fader, stylized as The FADER, is a New York City-based music magazine.
  Fact 1.1: The Fader was launched in 1999.
  Fact 1.2: The Fader was launched by Rob Stone and Jon Cohen.
  Fact 1.3: The Fader covers music, style and culture.
Sentence 2:  It was the first print publication to be released on iTunes.
  Fact 2.0: The Fader was the first print publication to be released on iTunes.

Retrieved items:
<item 1>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 1>

<item 2>
Audacious (CupcakKe album)
In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
</item 2>

<item 3>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
</item 3>

<item 4>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 4>

<item 5>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 5>

<item 6>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 6>

<item 7>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 7>

<item 8>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 8>

<item 9>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 9>

<item 10>
State (magazine)
State.ie (formerly State Magazine) is an Irish website and formerly a monthly music magazine, which launched in April 2008 and ceased to print in January 2009
having published a total of ten issues.

 The magazine continues online and was voted Best Music Site in 2008 and Best Web Publication in 2010 in the Irish Web Awards.

 Originally the concept of the magazine involved a hard copy of which there was a charge to purchase, however after six issues it was decided to make the
magazine's content free both online and in print.

 The first issue, April 2008, appeared on Irish shelves on 6 March 2008 and featured Michael Stipe of R.E.M. on the cover.

 This immediately garnered comparisons between the new magazine's similarities with "Hot Press" who featured Stipe on their cover at the same time, a move
widely thought to be an attempt by Hot Press to stifle State's status as a serious 'alternative' to the more established local magazine.

 At a price of €5.50, "State" charged €2 more than "Hot Press".
</item 10>
=== END ITEM id=6aa6230604951d6e ===

=== ITEM id=503eec544b3c011a ===
Gold sentences and facts:
Sentence 0: Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.
  Fact 0.0: Jean-François André Clervoy was born on 19 November 1958.
  Fact 0.1: Jean-François André Clervoy is a French engineer.
  Fact 0.2: Jean-François André Clervoy is a CNES astronaut.
  Fact 0.3: Jean-François André Clervoy is an ESA astronaut.
Sentence 1: Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
  Fact 1.0: Prof. Dr. Ulrich Hans Walter was born on February 9, 1954.
  Fact 1.1: Prof. Dr. Ulrich Hans Walter is a German physicist.
  Fact 1.2: Prof. Dr. Ulrich Hans Walter is a German engineer.
  Fact 1.3: Prof. Dr. Ulrich Hans Walter is a former DFVLR astronaut.

Retrieved items:
<item 1>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 1>

<item 2>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 2>

<item 3>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 3>

<item 4>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 4>

<item 5>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 5>

<item 6>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 6>

<item 7>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 7>

<item 8>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 8>

<item 9>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 9>

<item 10>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 10>
=== END ITEM id=503eec544b3c011a ===

=== ITEM id=36134aadfd76a93b ===
Gold sentences and facts:
Sentence 0: Akademisches Kunstmuseum (English:Academic Art Museum) is an art museum in Bonn, Germany.
  Fact 0.0: Akademisches Kunstmuseum is also known in English as the Academic Art Museum.
  Fact 0.1: Akademisches Kunstmuseum is an art museum.
  Fact 0.2: Akademisches Kunstmuseum is located in Bonn, Germany.
Sentence 1: The Federal City of Bonn (] ) is a city on the banks of the Rhine in the German state of North Rhine-Westphalia, with a population of over 300,000.
  Fact 1.0: The Federal City of Bonn is a city.
  Fact 1.1: The Federal City of Bonn is on the banks of the Rhine.
  Fact 1.2: The Federal City of Bonn is in the German state of North Rhine-Westphalia.
  Fact 1.3: The Federal City of Bonn has a population of over 300,000.

Retrieved items:
<item 1>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 1>

<item 2>
Alutiiq Museum
This cultural center features a gallery, storage for more than 190,000 local artifacts, including faunal materials, ethnobotanical samples, sediment samples,
field notes, photographs, and maps and a research laboratory.

 The Alutiiq Museum is a small repository, but we care for a very large collection with nearly 250,000 items.

 As a newly founded institution, the Alutiiq Museum sought to develop its policies and practices in professional ways.

 The Alutiiq Museum also uses its collections for community-building among the Alutiiq.

 The Alutiiq Museum is supported and governed by the Alutiiq Heritage Foundation and is dedicated to preserving and sharing Alutiiq heritage.
</item 2>

<item 3>
National Museum, Kraków
The National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), popularly abbreviated as MNK, established in 1879, is the main branch of Poland's
National Museum, which has several independent branches with permanent collections around the country.

 The Museum consists of 21 departments which are divided by art period; 11 galleries, 2 libraries, and 12 conservation workshops.

 It holds some 780,000 art objects, spanning from classical archeology to modern art, with special focus on Polish painting.

<ref name="culture.pl/muzea"> </ref><ref name="muzeum/Zbiory"> </ref>
</item 3>

<item 4>
Salzkammergut
The Salzkammergut is a resort area located in Austria.

 It stretches from the City of Salzburg eastwards along the Austrian Alpine Foreland and the Northern Limestone Alps to the peaks of the Dachstein Mountains,
spanning the federal states of Upper Austria, Salzburg, and Styria.

 The main river of the region is the Traun, a right tributary of the Danube.

 The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the
precious salt mines of the Habsburg Monarchy.

 It is a UNESCO World Heritage Site.
</item 4>

<item 5>
Croatian Natural History Museum
The Croatian Natural History Museum (Croatian: "Hrvatski prirodoslovni muzej" ) is the oldest and biggest natural history museum and the main body for natural
history research, preservation and collection in Croatia.

 Located on Dimitrije Demeter Street in Gornji Grad, one of the oldest neighbourhoods of the Croatian capital Zagreb, it owns one of the biggest museum
collections in Croatia, with over 2 million artefacts, including over 1.1 million animal specimens.

 It was founded in 1846 as the "National Museum".

 The National Museum was later split up into five museums, three of which were in 1986 merged as departments of the newly named Croatian Natural History Museum.

 The museum contains a large scientific library open to the public, and publishes the first Croatian natural history scientific journal, "Natura Croatica".
</item 5>

<item 6>
Akademisches Kunstmuseum
Akademisches Kunstmuseum (English:Academic Art Museum) is an art museum in Bonn, Germany.

 It is the oldest museum in Bonn and houses the antique collection of the University of Bonn with more than 500 antique statues and reliefs, and over 2,000
originals.

 It is located in a neoclassical building at the southern end of the Hofgarten, near the Electoral Palace.
</item 6>

<item 7>
Alutiiq Museum
The Alutiiq Museum or Alutiiq Museum and Archaeological Repository is a non-profit museum and cultural center dedicated to preserving and sharing the cultural
traditions of the Koniag Alutiiq branch of Sugpiaq ~ Alutiiq of the Alaska Native people.

 The museum is a state-of-the-art facility and is located on the first floor of the "Alutiiq Center" of Kodiak, Alaska.

 Alutiiq Museum is one of three museums in Kodiak.

 The museum is the seventh museum in Alaska and the second "tribal museum" in the United States to be accredited.

 The museum provides tours of its exhibits, laboratory and collections storage facilities to educational groups.

 The museum will accept materials relevant to the prehistoric, historic, and contemporary cultural history of the Native peoples who settled the Koniag Alutiiq
Nation.

 Such materials include, but are not limited to, archaeological, ethnological, photographic, film, audio, archival, and natural history specimens.
</item 7>

<item 8>
Animal Demography Unit
The Animal Demography Unit (ADU) is a formally recognized research unit of the University of Cape Town (UCT) located within the Department of Biological
Sciences of UCT.

 (The Department of Biological Sciences was formed from the merger of the Department of Botany and the Department of Zoology at the start of the 2013 academic
year).

 The Animal Demography Unit, popularly known as the ADU, was responsible for the management of the First and Second Southern African Bird Atlas Projects SABAP1
and SABAP2.

 The unit has submitted over eight million georeferenced biodiversity records to GBIF.
</item 8>

<item 9>
National Museum of Poland
"National Museum of Poland" is the common name for several of the country's largest and most notable museums. Poland's National Museum comprises several
independent branches, each operating a number of smaller museums.

 The main branch is the National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), established in 1879 with permanent collections consisting of several
hundred thousand items – kept in big part at the Main Building (along the 3 Maja St.), but also in the eight of its divisions around the city.
</item 9>

<item 10>
Bonn
The Federal City of Bonn (] ) is a city on the banks of the Rhine in the German state of North Rhine-Westphalia, with a population of over 300,000.

 About 24 km south-southeast of Cologne, Bonn is in the southernmost part of the Rhine-Ruhr region, Germany's largest metropolitan area, with over 11 million
inhabitants.
</item 10>
=== END ITEM id=36134aadfd76a93b ===

=== ITEM id=a60d25a9bdc7f7f1 ===
Gold sentences and facts:
Sentence 0:  Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
  Fact 0.0: Her personal best throw is 58.87 metres.
  Fact 0.1: She achieved her personal best throw at the 2002 Asian Games.
  Fact 0.2: The 2002 Asian Games were held in Busan.
Sentence 1:  A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.
  Fact 1.0: A total of 460 athletes took part in the competition.
  Fact 1.1: The athletes comprised 281 men and 179 women.
  Fact 1.2: The athletes came from 39 nations.

Retrieved items:
<item 1>
Masayoshi Yamamoto
Masayoshi Yamamoto (Japanese: 山本 雅賢) is a Japanese male artistic gymnast, representing his nation at international competitions.

 He won the gold medal in the Pommel horse event at the 2014 Asian Games in Incheon, South Korea.
</item 1>

<item 2>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 2>

<item 3>
Sinimole Paulose
Sinimol Paulose (born 24 June 1983) is an Indian middle distance runner.

 Her favorite events are 800 metres and 1500 metres.

 In 2009, she received Arjuna Award for her contribution to the Indian athletics.

 Sinimol was born in Piravom in Ernakulam district of Kerala state.

 She represented India at Beijing Olympics.
</item 3>

<item 4>
Daniel Ståhl
In June 2017, Ståhl managed a throw of 71.29 metres in Sollentuna, setting a new personal best and improving the Swedish record set in 1984 by Rickard Bruch by
three centimetres.

 Ståhl's throw was the farthest discus throw in the world for four years.
</item 4>

<item 5>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 5>

<item 6>
Cleopatra Borel
Cleopatra Borel (from 2005 until 2010 Borel-Brown; born 10 March 1979) is a female shot putter from Plaisance, Mayaro, Trinidad and Tobago.

 Her personal best throw is 19.42 metres, achieved in July 2011 at the Paris Diamond League Meeting.

 She has a personal best of 19.48 metres on the indoor track, achieved in February 2004 in Blacksburg.
</item 6>

<item 7>
Pavel Lyzhyn
Pavel Kanstantsinavich Lyzhyn (Belarusian: Павел Канстанцінавіч Лыжын, Pavieł Łyžyn , born 24 March 1981 in Vysokaye) is a Belarusian shot putter.

 His personal best throw is 21.21 metres from 2010.

 He threw a personal best throw of 20.98 metres at the 2008 Olympics in Beijing which originally translated into a fourth place, just 5 cm short of the bronze
medal winner and 11 cm short of the silver.

 However, on 25 November 2016 the IOC disqualified him from the 2008 Olympic Games and struck his results from the record for failing a drugs test in a
re-analysis of his doping sample from 2008.
</item 7>

<item 8>
Lee Young-sun
Lee Young-Sun (born 21 February 1974) is a retired female javelin thrower from South Korea.

 Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
</item 8>

<item 9>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 9>

<item 10>
Zhao Qinggang
Zhao Qinggang (; born 24 July 1985) is a Chinese track and field athlete who competes in the javelin throw.

 His 89.15 m personal best throw, set in 2014, is the Chinese record.

 It was also the Asian record until surpassed in 2017.

 He represented his country at the 2013 World Championships in Athletics.

 He is the 2014 Asian Games champion.

 At the East Asian Games he was the runner-up in 2009 and winner in 2013.

 He is a two-time Chinese champion (2012 and 2013) and was the winner of the 2013 National Games of China.
</item 10>
=== END ITEM id=a60d25a9bdc7f7f1 ===

=== ITEM id=f888d54ad113e2b1 ===
Gold sentences and facts:
Sentence 0: "Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".
  Fact 0.0: 'Into Dust' is a song by the American duo Mazzy Star.
  Fact 0.1: 'Into Dust' appears as the ninth track on the second album of Mazzy Star.
  Fact 0.2: The second album of Mazzy Star is 'So Tonight That I Might See'.
Sentence 1: Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and bassist Kendra Smith.
  Fact 1.0: Mazzy Star is an American alternative rock band.
  Fact 1.1: Mazzy Star was formed in Santa Monica, California, in 1989.
  Fact 1.2: Mazzy Star was formed from the group Opal.
  Fact 1.3: Opal was a collaboration of guitarist David Roback and bassist Kendra Smith.

Retrieved items:
<item 1>
Mazzy Star
Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and
bassist Kendra Smith.

 Roback's friend Hope Sandoval became the group's vocalist when Smith left the band.
</item 1>

<item 2>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 2>

<item 3>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 3>

<item 4>
Berlin (band)
Berlin is an American new wave band.

 The group was formed in Orange County in 1979 by John Crawford (bass guitar).

 Band members included Crawford, Terri Nunn (vocals), David Diamond (keyboards), Ric Olsen (guitar), Matt Reid (keyboards) and Rod Learned (drums).

 The band gained mainstream-commercial success in the early 1980s with singles including "The Metro", "Sex (I'm A..

.)

", "No More Words" and then in the mid 1980s with the chart-topping single "Take My Breath Away" from the 1986 film "Top Gun".
</item 4>

<item 5>
Into Dust
"Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".

 The song's arrangement is sparse and features only acoustic guitar and violin as accompaniment to the vocals.
</item 5>

<item 6>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 6>

<item 7>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 7>

<item 8>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 8>

<item 9>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 9>

<item 10>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 10>
=== END ITEM id=f888d54ad113e2b1 ===

=== ITEM id=bace83560582b534 ===
Gold sentences and facts:
Sentence 0: Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.
  Fact 0.0: Pinhead Gunpowder is an American punk rock band.
  Fact 0.1: Pinhead Gunpowder formed in East Bay, California, in 1990.
Sentence 1: Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.
  Fact 1.0: Billie Joe Armstrong was born on February 17, 1972.
  Fact 1.1: Billie Joe Armstrong is an American musician, singer, songwriter and actor.
  Fact 1.2: Billie Joe Armstrong is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day.
  Fact 1.3: Billie Joe Armstrong co-founded Green Day with Mike Dirnt.

Retrieved items:
<item 1>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 1>

<item 2>
Green Day related projects
Since 1991, some members of punk band Green Day have branched out past their "main band" and have started other projects with other musicians and have released
full-length albums and several EPs.

 Notable related projects of Green Day include Billie Joe Armstrong's Pinhead Gunpowder (which also featured Green Day's other guitarist Jason White), The
Frustrators in which Mike Dirnt plays bass, and The Network which many speculate has all three members of Green Day, although under stagenames.
</item 2>

<item 3>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 3>

<item 4>
Graph query row:
group: Pinhead Gunpowder
kind: punk rock band
relationship: MEMBER_OF
person: Billie Joe Armstrong
</item 4>

<item 5>
Bill Schneider (musician)
William 'Bill' Schneider (born June 17) is an American musician.

 He has played bass in the punk rock band Pinhead Gunpowder since 1990, as well as providing vocals and writing the lyrics to the song "Backyard Flames".

 He has also been in many other bands such as The Influents, Monsula and Uranium 9v, Sawhorse and The Skinflutes.

 Bill was worked as bass tech during the Nimrod era for Mike Dirnt, Green Day's and as guitar tech on their albums "Nimrod" and "Warning", American Idiot, 21st
Century Breakdown, Uno Dos Tres, Revolution Radio and was their day to day manager from 2003-2012 for their "American Idiot" and 21st Century Breakdown tours
through Uno Dos Tres.

 He also owned "Broken Guitars", a guitar shop in Oakland, California with Billie Joe Armstrong.

 Bill Still works on the sidelines with the band as well as in the studio.
</item 5>

<item 6>
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 6>

<item 7>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 7>

<item 8>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 8>

<item 9>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 9>

<item 10>
Carry the Banner
Carry the Banner is the third EP by the Berkeley, California-based punk rock band Pinhead Gunpowder.

 Originally released on 10" vinyl in December 1994 through Too Many Records, the EP was reissued on CD by Lookout Records shortly after as the initial vinyl
pressing sold out quickly.

 It was the group's first release to feature Jason White on guitar/vocals, replacing Sarah Kirsch, who left the band in 1994 due to differences with Billie Joe
Armstrong after his main band Green Day signed to major label Reprise Records.
</item 10>
=== END ITEM id=bace83560582b534 ===

=== ITEM id=3e8e264160306567 ===
Gold sentences and facts:
Sentence 0: "Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".
  Fact 0.0: 'Empty Orchestra' is the fourth episode of the third series of Inside No. 9.
  Fact 0.1: Inside No. 9 is a British dark comedy anthology television programme.
Sentence 1:  Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.
  Fact 1.0: 'Empty Orchestra' was written by Steve Pemberton.
  Fact 1.1: 'Empty Orchestra' was written by Reece Shearsmith.
  Fact 1.2: 'Empty Orchestra' was directed by Guillem Morales.
  Fact 1.3: 'Empty Orchestra' first aired on 7 March 2017.
  Fact 1.4: 'Empty Orchestra' first aired on BBC 2.
Sentence 2:  The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).
  Fact 2.0: The episode 'Empty Orchestra' is set in a karaoke booth.
  Fact 2.1: The episode 'Empty Orchestra' follows a group of work colleagues.
  Fact 2.2: The work colleague Greg is shown as Shearsmith.
  Fact 2.3: The work colleague Fran is shown as Sarah Hadland.
  Fact 2.4: The work colleague Connie is shown as Tamzin Outhwaite.
  Fact 2.5: The work colleague Janet is shown as Emily Howlett.
  Fact 2.6: The work colleague Duane is shown as Javone Prince.
  Fact 2.7: The colleagues are celebrating the promotion of Roger.
  Fact 2.8: Roger is shown as Pemberton.
Sentence 3:  Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and television productions, including army series "Red Cap" and crime drama "New Tricks".
  Fact 3.0: The performer described in this sentence came to national notice playing Melanie Owen in the BBC One soap opera EastEnders from 1998 until 2002.
  Fact 3.1: The performer has since starred in theatre productions.
  Fact 3.2: The performer has since starred in television productions.
  Fact 3.3: The performer has starred in the army series Red Cap.
  Fact 3.4: The performer has starred in the crime drama New Tricks.

Retrieved items:
<item 1>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 1>

<item 2>
Tamzin Outhwaite
Tamzin Maria Outhwaite ( ; born 5 November 1970) is a British actress from London.

 Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and
television productions, including army series "Red Cap" and crime drama "New Tricks".
</item 2>

<item 3>
Graph query row:
actor: Tamzin Outhwaite
role: Connie
work: Empty Orchestra
air_date: 7 March 2017
</item 3>

<item 4>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 4>

<item 5>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 5>

<item 6>
Empty Orchestra
"Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".

 Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.

 The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily
Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).

 Rebekah Hinds also stars.
</item 6>

<item 7>
Out of Control (2002 film)
Out of Control is a 2002 British television film written and directed for BBC by Dominic Savage.

 It stars Tamzin Outhwaite as Shelley Richards, an impoverished single mother whose son, Dean (Danny Young), is involved in a string of crimes that lead to his
incarceration in a young offenders institute (YOI).

 David Morrissey plays prison warden Mike, who tries to keep Dean out of trouble but has difficultly watching him all the time.

 After continuous bullying at the YOI, Dean is no longer able to cope and makes a fatal decision.

 It is the third in a loose trilogy of films by Savage about social deprivation, following "Nice Girl" (2000) and "When I Was Twelve" (2001).
</item 7>

<item 8>
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 8>

<item 9>
Melanie Owen
Melanie Jane "Mel" Owen (also Healy and Beale) is a fictional character from the BBC soap opera "EastEnders", played by Tamzin Outhwaite.

 She appeared from 19 October 1998 to 12 April 2002 when Outhwaite opted to leave the role.

 Mel was introduced by executive producer Matthew Robinson and became a prominent character during her tenure in the soap.
</item 9>

<item 10>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 10>
=== END ITEM id=3e8e264160306567 ===

=== ITEM id=d231bc54f813d3b3 ===
Gold sentences and facts:
Sentence 0:  She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.
  Fact 0.0: The woman described is married to Scott Stapp.
  Fact 0.1: Scott Stapp is a former frontman of the band Creed.
  Fact 0.2: Scott Stapp is the current frontman for the band Art of Anarchy.
Sentence 1:  The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between members.
  Fact 1.0: The band's third album, “Weathered”, was released in 2001.
  Fact 1.1: Tremonti handled bass when “Weathered” was released.
  Fact 1.2: The band disbanded in 2004.
  Fact 1.3: The band disbanded in 2004 due to increasing tension between members.

Retrieved items:
<item 1>
Jaclyn Stapp
Jaclyn Nesheiwat Stapp (born July 29, 1980) is a beauty queen, philanthropist and fashion model with pageant roots in Florida and New York.

 She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.

 Her most notable titles include Mrs. Florida America 2008 and Miss New York USA 2004.

 She is executive director of The Scott Stapp With Arms Wide Open Foundation.
</item 1>

<item 2>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 2>

<item 3>
A Compilation of Scott Weiland Cover Songs
A Compilation of Scott Weiland Cover Songs is a compilation album and third solo studio album by Stone Temple Pilots frontman Scott Weiland, released on August
30, 2011.

 The album consists entirely of cover songs, featuring an array of covers of artists that inspired Weiland, such as David Bowie, The Beatles, The Rolling
Stones, and The Smiths.

 The album was originally set to be released together with Weiland's memoir "Not Dead & Not for Sale" but Weiland decided later to release the album by itself
exclusively in digital format.
</item 3>

<item 4>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 4>

<item 5>
Creedence Clearwater Revival: Box Set
Creedence Clearwater Revival: Box Set is a box set by Creedence Clearwater Revival, released in 2001.

 It contains all of their complete studio albums, two complete live albums, and material recorded by the band under their previous names "The Golliwogs" and
"The Blue Velvets".

 In November 2013 the box set was reissued with different artwork.

 This box set spans their career, and contains all of their complete materials.
</item 5>

<item 6>
Scott Weiland and the Wildabouts
Scott Weiland and the Wildabouts (sometimes billed simply as The Wildabouts) were a rock and roll band formed sometime around 2012 by former Stone Temple Pilots
and Velvet Revolver lead singer Scott Weiland, along with bassist Tommy Baker, guitarist Jeremy Brown, and multi-instrumentalist Doug Grean.

 The Wildabouts were marred with several roster changes (including the death of guitarist Jeremy Brown) and ultimately disbanded after Weiland's sudden death on
a tour bus in late 2015.
</item 6>

<item 7>
Creed (band)
Creed is an American rock band formed in 1993 in Tallahassee, Florida.

 The band's best-known line-up consists of lead vocalist Scott Stapp, guitarist and vocalist Mark Tremonti, bassist Brian Marshall, and drummer Scott Phillips.

 Creed released two studio albums, "My Own Prison" in 1997 and "Human Clay" in 1999, before Marshall left the band in 2000.

 The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between
members.

 Tremonti, Marshall, and Phillips went on to found Alter Bridge while Stapp followed a solo career.
</item 7>

<item 8>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 8>

<item 9>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 9>

<item 10>
Creedence Clearwater Revival (album)
Creedence Clearwater Revival is the debut studio album by the American rock band Creedence Clearwater Revival, released in 1968.
</item 10>
=== END ITEM id=d231bc54f813d3b3 ===

=== ITEM id=e381b045d2e90fce ===
Gold sentences and facts:
Sentence 0:  It is found in Yemen and India.
  Fact 0.0: The subject of this sentence is found in Yemen.
  Fact 0.1: The subject of this sentence is found in India.
Sentence 1:  Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.
  Fact 1.0: Yemen's capital has been temporarily relocated to the port city of Aden.
  Fact 1.1: Aden is on the southern coast.
  Fact 1.2: The relocation is stated to be because of the content of the previous sentence.

Retrieved items:
<item 1>
Agrotis orthogonia
The Pale Western Cutworm ("Agrotis orthogonia") is a moth of the Noctuidae family.

 It is found in North America, more specifically dry, semi-desert areas of western North America from southern Canada to California, ranging eastward nearly to
the eastern edge of the Great Plains.
</item 1>

<item 2>
Ornithodoros savignyi
Because of its habit of feeding and dropping from its host, adult dispersal is limited, whereas larvae may remain attached to their hosts for several days.

 During its life cycle it will feed on multiple hosts between moults.
</item 2>

<item 3>
Aulacodes templalis
Aulacodes templalis is a moth in the Crambidae family.

 It was described by Schaus in 1906.

 It is found in Brazil.
</item 3>

<item 4>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 4>

<item 5>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
</item 5>

<item 6>
Yemen
Yemen ( ; Arabic: اليَمَن‎ ‎ "al-Yaman "), officially known as the Republic of Yemen (الجمهورية اليمنية "al-Jumhūrīyah al-Yamanīyah "), is an Arab country in
Western Asia at the southern end of the Arabian Peninsula.

 Yemen is the second-largest country in the peninsula, occupying 527,970 km (203,850 sq mi).

 The coastline stretches for about 2,000 km (1,200 mi).

 It is bordered by Saudi Arabia to the north, the Red Sea to the west, the Gulf of Aden and Arabian Sea to the south, and Oman to the east-northeast.

 Although Yemen's constitutionally stated capital is the city of Sana'a, the city has been under rebel control since February 2015.

 Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.

 Yemen's territory includes more than 200 islands; the largest of these is Socotra.
</item 6>

<item 7>
Ornithodoros savignyi
Ornithodoros savignyi aka Sand tampan or African eyed tampan or Kalahari sand tampan, is one of some 37 species in the genus "Ornithodoros" and is a soft tick
with a leathery, mammillated integument, causing paralysis and tampan toxicosis, two unrelated conditions.

 The sand tampan is an ectoparasite on humans, their livestock and wild animals, including birds and bats.

 Occurring in semi-desert areas of Africa, Saudi Arabia and other parts of the Persian Gulf, India, Sri Lanka and into Asia, it is able to survive for lengthy
periods without feeding, spending most of its life burrowed under sand or loose soil, often in wait for animals that rest or sleep under trees or in the lee of
rocks, but also in places where people or their animals congregate such as marketplaces, places of worship, cattle kraals and village squares.

 The timing of its activity is geared to coincide with that of potential hosts, but hot sunny conditions are usually avoided.
</item 7>

<item 8>
Amka
Amka (Hebrew: עַמְקָה‎ ), also known in Arabic as Amqa (Arabic: عمقا‎ ‎ ), is a moshav in the Matte Asher Regional Council of Israel's Northern District, near
Acre.

 The location of the moshav roughly corresponds the former Arab village, depopulated during the 1948 Arab–Israeli War.

 Yemenite Jews, who arrived from the southern Arab country of Yemen, founded the village's successor Amka in 1949.

 In 2016 its population was 706 .
</item 8>

<item 9>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 9>

<item 10>
Al Khor (city)
Al Khor is a coastal city in northern Qatar, located 50 kilometres north of the capital, Doha.

 It is the capital city of the municipality of Al Khor.

 The name of the city means creek in Arabic as the town is located on a creek.

 Al Khor is home to many employees of the oil industry due to its proximity to Qatar's northern oil and natural gas fields, and due to its proximity to the Ras
Laffan Industrial City.
</item 10>
=== END ITEM id=e381b045d2e90fce ===

=== ITEM id=590f647f356eba95 ===
Gold sentences and facts:
Sentence 0:  They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.
  Fact 0.0: This team is most well known for including Sugar Rashad Evans.
  Fact 0.1: Sugar Rashad Evans is a former UFC light heavyweight champion.
  Fact 0.2: This team is most well known for including Vitor Belfort.
  Fact 0.3: Vitor Belfort is a former UFC light heavyweight champion.
  Fact 0.4: This team is most well known for including Eddie Alvarez.
  Fact 0.5: Eddie Alvarez is a former UFC lightweight champion.
Sentence 1: Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
  Fact 1.0: Vitor Vieira Belfort was born on April 1, 1977.
  Fact 1.1: Vitor Vieira Belfort is a Brazilian mixed martial artist.
  Fact 1.2: Vitor Vieira Belfort is currently competing in the Middleweight division of the Ultimate Fighting Championship.

Retrieved items:
<item 1>
Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 He is the former UFC Light Heavyweight Champion, UFC 12 Heavyweight Tournament Champion, and the Cage Rage World Light Heavyweight Champion.

 As of June 16, 2017, he is the #11 contender in official UFC Middleweight rankings.
</item 1>

<item 2>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 2>

<item 3>
Luke Barnatt
Luke Barnatt (born 13 April 1988) is an English mixed martial artist who fought as a Middleweight for the Ultimate Fighting Championship.

 He was a member of FX's , and has also competed for BAMMA and UCMMA in England.

 At 6' 6" Barnatt is one of the tallest fighters currently competing in his weight class.

 Barnatt received his nickname "Bigslow" from brothers John Maguire and Tommy Maguire, for his size and being slow in response to their jokes and insults.

 Barnatt competes in the middleweight division of the Absolute Championship Berkut.
</item 3>

<item 4>
David Branch (fighter)
David Branch (born September 26, 1981) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2007, he has also formerly competed for the World Series of Fighting, Bellator, Shark Fights, and Titan FC.

 Branch was the inaugural World Series of Fighting Light Heavyweight and Middleweight Champion.

 He is currently ranked as the #9 Middleweight in the world according to Sherdog.
</item 4>

<item 5>
Kelvin Gastelum
Kelvin Gastelum (born October 24, 1991) is an American professional mixed martial artist who is currently signed to the Ultimate Fighting Championship (UFC).

 He was the winner of "The Ultimate Fighter 17".

 Gastelum is currently competing as a middleweight and is ranked #8 in the UFC official middleweight rankings.
</item 5>

<item 6>
Sam Alvey
Samuel Alvey (born May 6, 1986) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.

 A professional MMA competitor since 2008, Alvey has also formerly competed for Bellator, King of the Cage, the MFC, and was a contestant on .

 As of March 2017, Alvey is #14 in the official UFC Middleweight Rankings.
</item 6>

<item 7>
Blackzilians
The Blackzilians is a defunct professional team of fighters in mixed martial arts, boxing, kickboxing, amateur wrestling and catch wrestling located in South
Florida.

 They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC
lightweight champion Eddie Alvarez.
</item 7>

<item 8>
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 8>

<item 9>
Rashad Evans
Rashad Anton Evans (born September 25, 1979) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2003, Evans was the Heavyweight winner of The Ultimate Fighter 2 and is also a former UFC Light Heavyweight Champion.
</item 9>

<item 10>
Luke Zachrich
Luke Zachrich (born October 1, 1981) is an American mixed martial artist and former professional boxer who formerly competed in the Middleweight division of the
Ultimate Fighting Championship A professional mixed martial arts competitor since 2006, Zachrich was a member of Team Forrest on the seventh season of The
Ultimate Fighter, and has also fought for King of the Cage, Xtreme Fighting Organization, Bellator, and Ultimate Victory Challenge, the latter of which he was
the inaugural Middleweight Champion.
</item 10>
=== END ITEM id=590f647f356eba95 ===

=== ITEM id=932bbc8aa113a7c5 ===
Gold sentences and facts:
Sentence 0:  It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).
  Fact 0.0: The company began in 1995.
  Fact 0.1: The company began as the merger of two take-and-bake pizza companies.
  Fact 0.2: One of the merged companies was Papa Aldo's Pizza.
  Fact 0.3: Papa Aldo's Pizza was founded in 1981.
  Fact 0.4: The other merged company was Murphy's Pizza.
  Fact 0.5: Murphy's Pizza was founded in 1981.
Sentence 1: Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.
  Fact 1.0: Gino's Pizza and Spaghetti is a restaurant chain.
  Fact 1.1: Gino's Pizza and Spaghetti has 40 locations.
  Fact 1.2: Most of Gino's Pizza and Spaghetti's locations are within the U.S. state of West Virginia.
Sentence 2:  The company was founded by Kenney Grant in 1961.
  Fact 2.0: The company was founded by Kenney Grant.
  Fact 2.1: The company was founded in 1961.

Retrieved items:
<item 1>
Gino's East
Gino's East is a Chicago-based restaurant chain, notable for its deep-dish pizza (sometimes called Chicago-style pizza), and for its interior walls, which
patrons have covered in graffiti and etchings.

 The restaurant features deep-dish pizza baked in cast-iron pans, as well as sandwiches, soups and salads.
</item 1>

<item 2>
Nirula's
Nirula's is India's oldest fast food restaurant chain.

 Based in North India and most popular in NCR Delhi, it was Delhi's first fast food restaurant, opening in Connaught Place in 1977.

 Today it has over 70 outlets in NCR Delhi, Bihar, Haryana, Madhya Pradesh, Punjab, Rajasthan and Uttar Pradesh states, offering a “Desi” version of Western
fast food items.

 Nirula's success has led them to branch out into other ventures which include, ‘Potpourri’, an Indian cuisine, casual dining restaurant chain; ‘Nirula's 21’,
ice cream parlour chain, in addition to pastry shops and two hotels in Noida and Panipat.

 Recently Nirula's opened its first franchise in Patna, their first outlet in the entire east zone.
</item 2>

<item 3>
Papa Gino's
Papa Gino's, Inc. is a restaurant chain based in Dedham, Massachusetts specializing in American-style pizza along with pasta, subs, salads, and a variety of
appetizers.

 There are over 150 Papa Gino's locations in Connecticut, Maine, Massachusetts, New Hampshire, and Rhode Island.
</item 3>

<item 4>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 4>

<item 5>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 5>

<item 6>
Gino's Pizza and Spaghetti
Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.

 The company was founded by Kenney Grant in 1961.

 Many locations are shared with Tudor's Biscuit World although the Gino's brand is exclusive to West Virginia.

 There is one located in Ohio, while there are stand alone Tudor's locations in eastern Kentucky, southern Ohio and southwest Virginia.

 Gino's serves pizza, spaghetti, sandwiches, and more.

 Company headquarters are located in Huntington, West Virginia and Nitro, West Virginia.
</item 6>

<item 7>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 7>

<item 8>
Papa Murphy's Park
Papa Murphy's Park (formerly known as the Cal Expo Multi-Use Sports Field Facility and Bonney Field) is a sports venue located on the grounds of Cal Expo in
Sacramento, California.

 The soccer-specific stadium has a capacity of 11,569 and includes a full-sized (120 x 80 yard) soccer field.

 Papa Murphy's Park is the current home of Sacramento Republic FC soccer team and former home of PRO Rugby team, Sacramento Express.
</item 8>

<item 9>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 9>

<item 10>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 10>
=== END ITEM id=932bbc8aa113a7c5 ===

=== ITEM id=7a102387baf7ca15 ===
Gold sentences and facts:
Sentence 0: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 0.0: Warrel Dane was born on March 7, 1969.
  Fact 0.1: Warrel Dane is the lead singer of the band Sanctuary.
  Fact 0.2: Sanctuary is an American power metal band.
Sentence 1: Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.
  Fact 1.0: Roy Sætre Khantatat was born on March 12, 1970.
  Fact 1.1: Roy Sætre Khantatat is a Norwegian singer-songwriter.

Retrieved items:
<item 1>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 1>

<item 2>
Steve Alvord
Steven Lee Alvord (born October 2, 1964 in Bellingham, Washington) is a former professional American football defensive tackle and defensive end in the National
Football League and World League of American Football.

 In his four-year pro career he played for the St. Louis/Phoenix Cardinals of the NFL, and the Barcelona Dragons of the WLAF.

 Alvord played college football at Washington.

He has a cool daughter, happy birthday Katie.
</item 2>

<item 3>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 3>

<item 4>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 4>

<item 5>
Bo Welch
Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
</item 5>

<item 6>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 6>

<item 7>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 7>

<item 8>
Cody B. Ware
Cody Morgan Verdecias (born September 4,1988), better known by his stage name Cody B. Ware is an American Rapper, Songwriter and A&R from Queens, New York who
works on hip-hop and black metal.

 He is a member of the Queens-Based music collective World's Fair.

 Verdecias is a child of interracial parents and he is of half-Afro-Rican and half-Jewish descent.

 According to Verdecias, Bad Rabbits "blessed" him with the stage name Cody B. Ware after Retired Pro Wrestler Koko B. Ware.

 Verdecias says that growing up, his musical influences in hip-hop were Nas, Andre 3000, Big Pun, Khujo Goodie of Goodie Mob and he was also inspired by rock
artists such as Daryl Palumbo of Glassjaw and Jacob Bannon of Converge.
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
Roy Khan
Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.

 Commonly known as Roy Khan or simply Khan, he was the lead vocalist for the symphonic power metal band Kamelot from 1997 to his departure in 2011.

 He co-wrote most of Kamelot's songs with the band's guitarist and founder Thomas Youngblood during his tenure.

 Prior to joining Kamelot, Khan was lead singer for the Norwegian progressive metal band Conception from 1991 until 1997 when they disbanded.
</item 10>
=== END ITEM id=7a102387baf7ca15 ===
