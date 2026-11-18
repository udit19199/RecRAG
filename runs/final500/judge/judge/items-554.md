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

=== ITEM id=d19db804a084534d ===
Gold sentences and facts:
Sentence 0:  It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that position for three weeks.
  Fact 0.0: The song was released as the lead single from the fourth studio album of its act, "Elephant."
  Fact 0.1: The song was released in March 2003.
  Fact 0.2: The song reached number one on the Modern Rock Tracks.
  Fact 0.3: The song maintained its number one position on the Modern Rock Tracks for three weeks.
Sentence 1:  Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album charts.
  Fact 1.0: The album "Elephant" was released on April 1, 2003.
  Fact 1.1: The album "Elephant" was released on V2 Records.
  Fact 1.2: The album "Elephant" garnered near unanimous critical acclaim.
  Fact 1.3: The album "Elephant" garnered commercial success.
  Fact 1.4: The album "Elephant" received a nomination for Album of the Year at the 46th Grammy Awards in 2004.
  Fact 1.5: The album "Elephant" won Best Alternative Music Album at the 46th Grammy Awards in 2004.
  Fact 1.6: The album "Elephant" peaked at No. 6 on the US "Billboard" charts.
  Fact 1.7: The album "Elephant" topped the UK album charts.

Retrieved items:
<item 1>
Meg White
Megan Martha "Meg" White (born December 10, 1974) is an American drummer and occasional singer known for her work with Jack White in the Detroit rock duo The
White Stripes.

 On an impulse, she played on Jack's drums in 1997.

 The two decided to form a band and began performing two months later, calling themselves The White Stripes because of their last name and Meg's affinity for
peppermint candy.

 The band quickly became a Detroit underground favorite, before reaching national, then international fame.

 White has been nominated for various awards as a part of the White Stripes, and has received four Grammy Awards.
</item 1>

<item 2>
Premio Lo Nuestro 2003
Premio Lo Nuestro 2003 was the 15th anniversary of the awards.

 the show was hosted by Mexican presenters Marco Antonio Regil and Adal Ramones.

 Juanes, Thalía, Marc Anthony, Pilar Montenegro, Sin Bandera, Banda el Recodo and other Latin music greats gave electrifying performances.

 In the show, there was 36 awards winners with 135 nominations.

 In Pop genre, Awards was given for : Album of the Year, Best Male Artist, Best Female Artist, Best Group or duo, Best New Artist and Song of the Year.

 In Rock Genre : Best Rock Album and Best rock Performer of the Year.

 In Tropical genre : Best Tropical Album of the Year, Best Tropical Male Artist, Best Tropical Female Artist, Best Tropical Group or Duo of the Year, Best
Tropical New Artist, Tropical Song of the Year, Best Merengue Performance, Best Salsa Performance and Best Traditional Performance.

 Juanes was the biggest winner of night, took home four awards Best Pop Male Artist, Best Music Video, Best Rock Performance, and Pop Song of the Year .
</item 2>

<item 3>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 3>

<item 4>
Seven Nation Army
"Seven Nation Army" (also stylized as "7 Nation Army") is a song by American rock duo The White Stripes.

 It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that
position for three weeks.

 It also became the third best-performing song of the decade on the same chart.

 It was well received commercially as well, and won the Grammy Award for Best Rock Song.
</item 4>

<item 5>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 5>

<item 6>
The collection surveys music albums, artists, songs, and genres across rock, pop, metal, and electronic music. Highlights include Patty Griffin’s 2004 album
*Impossible Dream*; Edward Sharpe and the Magnetic Zeros’ 2013 album and 2010 single; the White Stripes’ rise, breakup, acclaimed *Elephant* (2003), and
Grammy-winning “Seven Nation Army”; and biographies of Meg White and Jack White. Other entries cover releases by Steve Winwood
</item 6>

<item 7>
Grammy Award for Best Female Rap Solo Performance
The Grammy Award for Best Female Rap Solo Performance was an honor presented to female recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958 and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 7>

<item 8>
The selections survey music history across patriotic songs and anthem protests; reggae, country, jazz, rock, pop, R&B, and electronic music; and radio programs,
awards, and chart milestones. They profile artists including Cedric Myton, Florence Welch, Steve Winwood, and George Thorogood, and highlight releases by IU,
S.H.E, BT, Icehouse, and others. Notable achievements include Grammy and award wins, platinum and multi-million sales, chart-topping singles,
</item 8>

<item 9>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 9>

<item 10>
Grammy Award for Best Male Rap Solo Performance
The Grammy Award for Best Male Rap Solo Performance was an honor presented to male recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958, and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 10>
=== END ITEM id=d19db804a084534d ===

=== ITEM id=473a25a1d2301aa1 ===
Gold sentences and facts:
Sentence 0: Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.
  Fact 0.0: Enrico Cocozza was born on 6 November 1921.
  Fact 0.1: Enrico Cocozza died on 27 December 1997.
  Fact 0.2: Enrico Cocozza was a Scottish filmmaker.
  Fact 0.3: Enrico Cocozza won many film awards during the 1940s and 1950s.
Sentence 1: ", which was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.0: A film was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.1: The film whose title is quoted in this fragment was nominated for the Academy Award for Best Foreign Language Film.

Retrieved items:
<item 1>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 1>

<item 2>
Eric Aghimien
Eric Enomamien Aghimien is a Nigerian director, producer, screenwriter and editor.

 His debut feature film, "A Mile from Home" won awards at both the 2014 Africa Magic Viewers Choice Awards and the 10th Africa Movie Academy Awards.
</item 2>

<item 3>
Enrico Cocozza
Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.

 His often surreal films were mainly filmed in and around the town of Wishaw in Scotland, where his family owned the popular Belhaven Cafe.

 These include "Chick's Day" (1950), a prize winner at the 1951 Scottish Amateur Film Festival, "The Living Ghost" (1957), and "Glasgow's Docklands" (1959).

 Illness later forced him to give up making films and he spent most of his working life teaching at the University of Strathclyde in Glasgow.

 In 2001 he was the subject of the Channel 4 documentary "Artery: the Story of Enrico Cocozza".
</item 3>

<item 4>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 4>

<item 5>
Maya Deren Award
The American Film Institute Award for Independent Film and Video Artists, subtitled and generally known as the Maya Deren Award, was an award presented to
filmmakers and video artists by the American Film Institute to honor independent filmmaking.

 Named for the avant-garde experimental film artist Maya Deren, it was given from 1986 through 1996.
</item 5>

<item 6>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 6>

<item 7>
Salaam Bombay!
Salaam Bombay!

 is a 1988 Hindi film directed by Mira Nair, and screenwritten by her longtime creative collaborator, Sooni Taraporevala.

 The film chronicles the day-to-day life of children living on the streets of Bombay, India's biggest city.

 It won the National Film Award for Best Feature Film in Hindi, the National Board of Review Award for Top Foreign Film, the Golden Camera and Audience Awards
at the Cannes Film Festival, and three awards at the Montréal World Film Festival.

 The film was India's second film submission to be nominated for the Academy Award for Best Foreign Language Film.

 The film was among the list of "The Best 1,000 Movies Ever Made" by "The New York Times".
</item 7>

<item 8>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 8>

<item 9>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 9>

<item 10>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 10>
=== END ITEM id=473a25a1d2301aa1 ===

=== ITEM id=78426f7718eadd78 ===
Gold sentences and facts:
Sentence 0: Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space as part of the Soviet Union's Interkosmos programme.
  Fact 0.0: Sigmund Werner Paul Jähn was born on 13 February 1937.
  Fact 0.1: Sigmund Werner Paul Jähn is a German cosmonaut.
  Fact 0.2: Sigmund Werner Paul Jähn is a pilot.
  Fact 0.3: In 1978 Sigmund Werner Paul Jähn became the first East German to fly in space.
  Fact 0.4: In 1978 Sigmund Werner Paul Jähn became the first German native to fly in space.
  Fact 0.5: The 1978 flight by Sigmund Werner Paul Jähn was part of the Soviet Union's Interkosmos programme.
Sentence 1: Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.
  Fact 1.0: Arne Christer Fuglesang was born on March 18, 1957.
  Fact 1.1: Arne Christer Fuglesang was born in Stockholm.
  Fact 1.2: Arne Christer Fuglesang is a Swedish physicist.
  Fact 1.3: Arne Christer Fuglesang is an ESA astronaut.
Sentence 2:  He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
  Fact 2.0: Arne Christer Fuglesang was first launched aboard the STS-116 Space Shuttle mission.
  Fact 2.1: That launch was on December 10, 2006, at 01:47 GMT.
  Fact 2.2: Arne Christer Fuglesang became the first Swedish citizen in space.

Retrieved items:
<item 1>
Eberhard Köllner
Eberhard Köllner (born 29 September 1939 in Stassfurt, Germany) was selected for Soyuz 31 as the backup for Sigmund Jähn.
</item 1>

<item 2>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 2>

<item 3>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 3>

<item 4>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 4>

<item 5>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 5>

<item 6>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 6>

<item 7>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 7>

<item 8>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 8>

<item 9>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 9>

<item 10>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 10>
=== END ITEM id=78426f7718eadd78 ===

=== ITEM id=9953a185c251a104 ===
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
David Draiman
David Michael Draiman (born March 13, 1973) is an American songwriter and the vocalist for the band Disturbed as well as for the band Device.

 Draiman is known for his distorted voice and percussive singing style.

 In November 2006, Draiman was voted number 42 on the "Hit Parader"’s ""Top 100 Metal Vocalists of All Time"".

 Draiman has written some of Disturbed's most successful singles, such as "Stupify", "Down with the Sickness", "Indestructible", and "Inside the Fire".
</item 1>

<item 2>
Chicago’s “25 or 6 to 4,” written by Robert Lamm and sung by Peter Cetera, was recorded in 1969 and became the band’s first U.S. top-five hit; Terry Kath plays
its wah-wah guitar solo. Kath, Chicago’s founding guitarist, singer, and songwriter, also leads “Free” and “Brand New Love Affair” (first half) and sings
“Wishing You Were Here.” His final album was 1977’s *
</item 2>

<item 3>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 3>

<item 4>
Device (metal band)
Device was an industrial metal band started by David Draiman, frontman of the heavy metal group Disturbed.

 Draiman was approached by Geno Lenardo, former guitarist of Filter, and together they started to work on new material.

 The two started work on a debut album in June 2012.

 The result of those sessions, "Device", was released on April 9, 2013.

 The first single and third track, "Vilify" was released to radio ahead of the album on February 19, 2013, alongside its first music video, directed by P. R.
Brown.

 The second single and the introductory track, "You Think You Know", was released on June 11, 2013, alongside its music video, directed once again by Brown.
</item 4>

<item 5>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 5>

<item 6>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 6>

<item 7>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 7>

<item 8>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 8>

<item 9>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 9>

<item 10>
Pantera
Pantera was an American heavy metal band from Arlington, Texas.

 The group was formed in 1981 by the Abbott brothers – drummer Vinnie Paul and guitarist Dimebag Darrell – along with lead vocalist Terry Glaze.

 Bassist Rex Brown joined the band the following year, replacing Tommy D. Bradford, who was the unofficial original.

 Having started as a glam metal band, Pantera released four albums during the 1980s.

 Looking for a new and heavier sound, Pantera replaced Glaze with Phil Anselmo in late 1986 and released "Power Metal" in 1988.

 With its fifth album, 1990's "Cowboys from Hell", Pantera introduced a groove metal sound.

 Pantera's sixth album, 1992's "Vulgar Display of Power", exhibited an even heavier sound. "

Far Beyond Driven" (1994) debuted at number one on the "Billboard" 200.
</item 10>
=== END ITEM id=9953a185c251a104 ===

=== ITEM id=510a89ff1fbcba55 ===
Gold sentences and facts:
Sentence 0: Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.
  Fact 0.0: Jack and Jill is an American magazine for children.
  Fact 0.1: Jack and Jill is published bimonthly.
  Fact 0.2: Jack and Jill is for children 7 to 12 years old.
  Fact 0.3: The title of Jack and Jill comes from the nursery rhyme of the same name.
Sentence 1: Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.
  Fact 1.0: Paper, stylized as PAPER, is a New York City-based magazine.
  Fact 1.1: Paper is an independent magazine.
  Fact 1.2: Paper focuses on fashion.
  Fact 1.3: Paper focuses on popular culture.
  Fact 1.4: Paper focuses on nightlife.
  Fact 1.5: Paper focuses on music.
  Fact 1.6: Paper focuses on art.
  Fact 1.7: Paper focuses on film.

Retrieved items:
<item 1>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 1>

<item 2>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 2>

<item 3>
Mighty Jill Off
Mighty Jill Off is a 2D platform independently developed freeware video game designed by Anna Anthropy, with art by James Harvey and music by Andrew Toups.

 It stars a submissive named Jill, who has a boot fetish and is forced to climb up a tower after her Queen kicks her down it as punishment.

 Jill does this by jumping and slowly descending over obstacles.

 Jill can be defeated in one hit by these obstacles, but will return to the last check point.

 The game serves as an homage to the 1986 arcade game "Mighty Bomb Jack".

 It had follow-ups, such as "Mighty Jill Off - Jill Off Harder Edition" and "Jill Off With One Hand".

 Jill made a cameo appearance in the 2010 video game "Super Meat Boy" as a playable character.
</item 3>

<item 4>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 4>

<item 5>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 5>

<item 6>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 6>

<item 7>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 7>

<item 8>
Jack and Jill (magazine)
Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.

 It features stories and educational activities.
</item 8>

<item 9>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 9>

<item 10>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 10>
=== END ITEM id=510a89ff1fbcba55 ===

=== ITEM id=54432f2f5bf239ed ===
Gold sentences and facts:
Sentence 0:  It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
  Fact 0.0: The described act consists of a duet.
  Fact 0.1: Dave Hemingway is a vocalist featured in the duet.
  Fact 0.2: Briana Corrigan is a vocalist featured in the duet.
Sentence 1: Briana Corrigan (born 30 May 1965) is a Northern Irish singer.
  Fact 1.0: Briana Corrigan was born on 30 May 1965.
  Fact 1.1: Briana Corrigan is a Northern Irish singer.

Retrieved items:
<item 1>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 1>

<item 2>
Takes a Little Time (Total Contrast song)
"Takes a Little Time" is a 1985 dance hit by British, techno-soul, duo, Total Contrast.

 Their debut American release went to number one on the U.S. dance chart for one week.

 "Takes a Little Time" did not make the Hot 100 but, peaked at number eighty on the R&B singles chart.
</item 2>

<item 3>
Just a Little Time
Just a Little Time is an album by Lynsey de Paul.

 It was originally released in 30 September 1994 on the Music Deluxe (catalogue number MSCD9) label but has since been released on the Tring International PLC
label (catalogue number JHD128) and on Arc Records/The Magic Collection (Catalogue number MEC 949080).

 All of the songs on the album are written or co-written by de Paul.

 The songs "Sugar Me", Getting a Drag", "Storm in a Teacup and "Dancing on a Saturday Night" are radical re-recordings of her earlier hits performed in an
updated and uptempo fashion.

 "Won't Somebody Dance with Me" is still performed as a ballad, but has been updated and extended with an extra refrain.

 Never before released songs on the album include the ballads "Words Don't Mean A Thing" and title track "Just a Little Time" as well as the more uptempo
"Instant Love" and "Now & Then".
</item 3>

<item 4>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 4>

<item 5>
Takes a Little Time (Amy Grant song)
"Takes a Little Time" was a maxi-single released in 1997 (see 1997 in music) to promote Amy Grant's album "Behind the Eyes", which was also released that year.

 "Takes a Little Time" included two songs from "Behind the Eyes", as well as a new version of Grant's 1982 Christian radio hit, "El Shaddai".

 The maxi-single was also an enhanced CD (ECD), meaning it had a video that consumers could watch when they put it in their home computer.

 The ECD portion of the CD contained a live acoustic version of the song After the Fire, which Grant later released on her 2003 album "Simple Things"; Grant
stated on Oprah that she had written the song for her mother.
</item 5>

<item 6>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 6>

<item 7>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 7>

<item 8>
Graph query row:
person: Briana Corrigan
nationality: Northern Irish
creative_work: None
work_connection: None
related_person: None
person_connection: None
</item 8>

<item 9>
Houston (Means I'm One Day Closer to You)
"Houston (Means I'm One Day Closer to You)" is a song written by Larry Gatlin and recorded by American country music group Larry Gatlin & the Gatlin Brothers
Band.

 It was released in September 1983 as the first single from the album ""Greatest Hits Vol.

 II"" then included to first track of ""Not Guilty"" (1984).

 "Houston (Means I'm One Day Closer to You)" was the group's third and last number one on the country chart.

 The single went to number one for two weeks and spent a total of fifteen weeks on the country chart.
</item 9>

<item 10>
There Ain't No Good Chain Gang
"There Ain't No Good Chain Gang" is a song written by Hal Bynum and Dave Kirby, and recorded by American country music artists Johnny Cash and Waylon Jennings.

 It was released in May 1978 as the second single from the album "I Would Like to See You Again".

 The song reached #2 on the "Billboard" Hot Country Singles & Tracks chart.
</item 10>
=== END ITEM id=54432f2f5bf239ed ===

=== ITEM id=d8eeae0f70d1989d ===
Gold sentences and facts:
Sentence 0: The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a unique numeric identifier, referred to as a "DUNS number" to a single business entity.
  Fact 0.0: The Data Universal Numbering System is abbreviated as DUNS or D-U-N-S.
  Fact 0.1: The Data Universal Numbering System is a proprietary system.
  Fact 0.2: The Data Universal Numbering System is developed by Dun & Bradstreet (D&B).
  Fact 0.3: The Data Universal Numbering System is regulated by Dun & Bradstreet (D&B).
  Fact 0.4: The Data Universal Numbering System assigns a unique numeric identifier to a single business entity.
  Fact 0.5: The unique numeric identifier is referred to as a DUNS number.
Sentence 1:  Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
  Fact 1.0: Dun & Bradstreet is often referred to as D&B.
  Fact 1.1: The company's database contains more than 265 million business records.
  Fact 1.2: The company's database contains business records worldwide.

Retrieved items:
<item 1>
Universal Numbering System
The Universal Numbering System is a dental notation system for associating information to a specific tooth, and is commonly used in the United Kingdom.
</item 1>

<item 2>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 2>

<item 3>
Data Universal Numbering System
The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a
unique numeric identifier, referred to as a "DUNS number" to a single business entity.

 It was introduced in 1963 to support D&B's credit reporting practice.

 It is standard worldwide.

 DUNS users include the European Commission, the United Nations, and the United States government.

 More than 50 global industry and trade associations recognize, recommend, or require DUNS.

 The DUNS database contains over 250 million entries for businesses throughout the world.
</item 3>

<item 4>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 4>

<item 5>
DFDS
DFDS is Northern Europe's largest shipping and logistics company.

 The company's name is an abbreviation of Det Forenede Dampskibs-Selskab (literally "The United Steamship Company").

 DFDS was founded in 1866, when C.F. Tietgen merged the three biggest Danish steamship companies of that day.
</item 5>

<item 6>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 6>

<item 7>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 7>

<item 8>
GeneDx
GeneDx is genetic testing company that was founded in 2000 by two scientists from the National Institutes of Health (NIH), Drs. Sherri Bale and John Compton.

 They specialize in testing for rare inherited disorders.

 In 2006, BioReference laboratories acquired GeneDx.

 Since then GeneDx has operated as a subsidiary of this parent company under the leadership of Drs. Bale and Compton.
</item 8>

<item 9>
Animal Demography Unit
The Animal Demography Unit (ADU) is a formally recognized research unit of the University of Cape Town (UCT) located within the Department of Biological
Sciences of UCT.

 (The Department of Biological Sciences was formed from the merger of the Department of Botany and the Department of Zoology at the start of the 2013 academic
year).

 The Animal Demography Unit, popularly known as the ADU, was responsible for the management of the First and Second Southern African Bird Atlas Projects SABAP1
and SABAP2.

 The unit has submitted over eight million georeferenced biodiversity records to GBIF.
</item 9>

<item 10>
Dental notation
Dental professionals, in writing or speech, use several different dental notation systems for associating information with a specific tooth.

 The three most common systems are the ISO System, Universal Numbering System, and Palmer notation method.

 The ISO system is used worldwide, and the Universal is used widely in the United States.

 The ISO System can be easily adapted to computerized charting.
</item 10>
=== END ITEM id=d8eeae0f70d1989d ===

=== ITEM id=33d0213090584e92 ===
Gold sentences and facts:
Sentence 0: Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with punctate stigmas on the female flowers.
  Fact 0.0: Lithocarpus is a genus in the beech family Fagaceae.
  Fact 0.1: Lithocarpus differs from "Quercus" in the erect spikes of insect-pollinated male flowers.
  Fact 0.2: Lithocarpus differs from "Quercus" in the short styles with punctate stigmas on the female flowers.
Sentence 1:  It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.
  Fact 1.0: Lithocarpus contains 17 species of shrubs and small trees.
  Fact 1.1: The Lithocarpus species are native from southern Florida to Mexico and South America.

Retrieved items:
<item 1>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 1>

<item 2>
Right Now Kapow
Right Now Kapow is an American animated sketch comedy television series produced by Warner Bros.

 Animation for Disney XD.

 It is the first collaboration between Warner Bros.

 Animation and Disney.

 The series premiered on September 19, 2016 and ended on May 31, 2017.

 The series was created by Justin Becker and Marly Halpern-Graser, who previously worked on the Cartoon Network series "Mad".

 Becker also worked on Adult Swim infomercials, and Halpern-Graser also worked on Cartoon Network's "DC Nation".

 The series follows Dog, Candy, Ice Cream, Diamond, Plant, and Moon going on new adventures everyday...and find themselves in mischief.

 Every episode of the series has 3 main parts in every 11-minute segment and other random shorts.

 The series has a style of humor similar to Cartoon Network's previous series, "Mad".
</item 2>

<item 3>
Lamiaceae
or Labiatae is a family of flowering plants commonly known as the mint or deadnettle family.

 Many of the plants are aromatic in all parts and include widely used culinary herbs, such as basil, mint, rosemary, sage, savory, marjoram, oregano, hyssop,
thyme, lavender, and perilla.

 Some species are shrubs, trees (such as teak), or, rarely, vines.

 Many members of the family are widely cultivated, not only for their aromatic qualities but also their ease of cultivation, since they are readily propagated
by stem cuttings.

 Besides those grown for their edible leaves, some are grown for decorative foliage, such as "Coleus".

 Others are grown for seed, such as "Salvia hispanica" (chia), or for their edible tubers, such as "Plectranthus edulis", "Plectranthus esculentus",
"Plectranthus rotundifolius", and "Stachys affinis" (Chinese artichoke).
</item 3>

<item 4>
Lithocarpus glaber
Lithocarpus glaber, the Japanese oak, is a tree species in the genus "Lithocarpus" found in Japan, China and Taïwan.
</item 4>

<item 5>
Lithocarpus crassinervius
Lithocarpus crassinervius is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 5>

<item 6>
Yucca gloriosa var. tristis
Yucca gloriosa" var.

 "tristis (syn.

 Yucca recurvifolia, "Yucca gloriosa" var.

 "recurvifolia"), known as curve-leaf yucca, curved-leaved Spanish-dagger or pendulous yucca, is a variety of "Yucca gloriosa".

 It is often grown as an ornamental plant, but is native to the southeastern United States.

 In contrast to "Y. gloriosa" var.

 "tristis", the leaves of "Y. gloriosa" var.

 "gloriosa" are hard stiff, erect and narrower.
</item 6>

<item 7>
Lithocarpus platycarpus
Lithocarpus platycarpus is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.

.
</item 7>

<item 8>
Lithocarpus kostermansii
Lithocarpus kostermansii is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 8>

<item 9>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 9>

<item 10>
Duranta erecta
Duranta erecta is a species of flowering shrub in the verbena family Verbenaceae, native from Mexico to South America and the Caribbean.

 It is widely cultivated as an ornamental plant in tropical and subtropical gardens throughout the world, and has become naturalized in many places.

 It is considered an invasive species in Australia, China, South Africa and on several Pacific Islands.

 Common names include golden dewdrop, pigeon berry, and skyflower.

 In Mexico, the native Nahuatl name for the plant is xcambocoché.

 In Tonga it is known as mavaetangi (tears of departure).
</item 10>
=== END ITEM id=33d0213090584e92 ===

=== ITEM id=72c2fbac21fbdb99 ===
Gold sentences and facts:
Sentence 0:  The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova Planta Decree of Majorca and Ibiza (28 November 1715).
  Fact 0.0: The body that was to become the Gran i General Consell was founded in 1249.
  Fact 0.1: The Consell was abolished on 22 July 1718.
  Fact 0.2: The Consell was abolished by Philip V.
  Fact 0.3: Philip V was empowered by the Nova Planta Decree of Majorca and Ibiza.
  Fact 0.4: The Nova Planta Decree of Majorca and Ibiza is dated 28 November 1715.
Sentence 1: Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January 1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
  Fact 1.0: Philip V's Spanish name is Felipe V.
  Fact 1.1: Philip V's French name is Philippe.
  Fact 1.2: Philip V's Italian name is Filippo.
  Fact 1.3: Philip V was born on 19 December 1683.
  Fact 1.4: Philip V died on 9 July 1746.
  Fact 1.5: Philip V was King of Spain from 1 November 1700 to 15 January 1724.
  Fact 1.6: Philip V abdicated in favour of his son Louis on 15 January 1724.
  Fact 1.7: Philip V reassumed the Spanish throne on 6 September 1724, upon his son's death.
  Fact 1.8: Philip V was King of Spain from 6 September 1724 until his death on 9 July 1746.

Retrieved items:
<item 1>
Philip V of Spain
Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January
1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
</item 1>

<item 2>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 2>

<item 3>
Duke of Bedford
Duke of Bedford (named after Bedford, England) is a title that has been created six times (for five distinct people) in the Peerage of England.

 The first and second creations came in 1414 in favour of Henry IV's third son, John, who later served as regent of France.

 He was made Earl of Kendal at the same time and was made Earl of Richmond later the same year.

 The titles became extinct on his death in 1435.

 The third creation came in 1470 in favour of George Neville, nephew of Warwick the Kingmaker.

 He was deprived of the title by Act of Parliament in 1478.

 The fourth creation came 1478 in favour of George, the third son of Edward IV.

 He died the following year at the age of two.

 The fifth creation came in 1485 in favour of Jasper Tudor, half-brother of Henry VI and uncle of Henry VII.

 He had already been created Earl of Pembroke in 1452.

 However, as he was a Lancastrian, his title was forfeited between 1461 and 1485 during the predominance of the House of York.
</item 3>

<item 4>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 4>

<item 5>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 5>

<item 6>
Pedro Luis de Borja Lanzol de Romaní
Pedro Luis de Borja Lanzol de Romaní, O.S.Io.Hieros.

 (1472–1511) was a Roman Catholic cardinal and cardinal-nephew and papal military leader.

 He received a wide variety of sinecures during the papacy of his great-uncle, Pope Alexander VI, but was exiled to Naples on the election of Borja rival Pope
Julius II.

 Borja also fought with the Knights Hospitaller in Jerusalem and Rhodes.
</item 6>

<item 7>
Duke of Bedford
He regained the earldom in 1485 when his nephew Henry VII came to the throne and was elevated to the dukedom the same year.

 He had no legitimate children and the titles became extinct on his death in 1495.
</item 7>

<item 8>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 8>

<item 9>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 9>

<item 10>
The entries survey monarchs, dynasties, and succession systems. Hawaiʻi’s Kalākaua, its last king, revived public hula and Hawaiian culture. Hanover ruled
Britain from 1714 until Victoria’s death; Hanover and Oldenburg branches remain linked to European royal families. Calenberg became Hanover’s core, while
Hanover’s Ernest Augustus inherited its throne in 1837 under Salic law. Other topics include Silesian Bernstadt, Tus
</item 10>
=== END ITEM id=72c2fbac21fbdb99 ===

=== ITEM id=38042405ea1b45d2 ===
Gold sentences and facts:
Sentence 0:  She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".
  Fact 0.0: The person is a member of the Game of Thrones stunt teams.
  Fact 0.1: The Game of Thrones stunt teams won two Screen Actors Guild Awards.
  Fact 0.2: The person has done stunt performances for films including Rogue One.
Sentence 1:  "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
  Fact 1.0: Rogue One follows a group of rebels.
  Fact 1.1: The rebels are on a mission to steal the plans for the Death Star.
  Fact 1.2: The Death Star is the Galactic Empire's superweapon.

Retrieved items:
<item 1>
Race to Nowhere
Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
</item 1>

<item 2>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 2>

<item 3>
The Runaways (film)
The Runaways is a 2010 American drama film about the 1970s rock band of the same name written and directed by Floria Sigismondi.

 It is based on the book "Neon Angel: A Memoir of a Runaway" by the band's original lead vocalist Cherie Currie.

 The film stars Dakota Fanning as Currie, Kristen Stewart as rhythm guitarist and vocalist Joan Jett, and Michael Shannon as record producer Kim Fowley.

 "The Runaways" depicts the formation of the band in 1975 and focuses on the relationship between Currie and Jett until Currie's departure from the band.

 The film grossed about $4.7 million worldwide and received generally favorable reviews from critics.
</item 3>

<item 4>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 4>

<item 5>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 5>

<item 6>
Frenemies (film)
Frenemies is a 2012 teen comedy-drama television film and anthology based on the novel of the same name by Alexa Young which premiered on Disney Channel.

 It features an ensemble cast starring Bella Thorne, Zendaya, Stefanie Scott, Nick Robinson, Mary Mouser and features Connor Price, Jascha Washington and Dylan
Everett.

 The film follows three pairs of teenage friends that go from friends to enemies and back again.

 The film was directed by Daisy Mayer and written by Dava Savel, Wendy Weiner, and Jim Krieg.

 The Disney Channel Original Movie premiered on January 13, 2012 in the United States and Canada.
</item 6>

<item 7>
Rachelle Beinart
Rachelle Amy Beinart (born 29 September 1984) is a British actress and stunt performer.

 She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".

 She has also acted in preschool children's shows as a suit actor, including the reboot of "Teletubbies".
</item 7>

<item 8>
Rogue One
Rogue One: A Star Wars Story, or simply Rogue One, is a 2016 American epic space opera film directed by Gareth Edwards.

 The screenplay by Chris Weitz and Tony Gilroy is from a story by John Knoll and Gary Whitta.

 It was produced by Lucasfilm and distributed by Walt Disney Studios Motion Pictures.

 It is the first installment of the "Star Wars" "Anthology" series, set immediately before the events of the original "Star Wars" film.

 The cast includes Felicity Jones, Diego Luna, Riz Ahmed, Ben Mendelsohn, Donnie Yen, Mads Mikkelsen, Alan Tudyk, Jiang Wen and Forest Whitaker.

 "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
</item 8>

<item 9>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 9>

<item 10>
Rachelle Henry
Rachelle Henry (born December 16, 2000) is an American teen actress and filmmaker.

 She played the role of Sandy Hobbs in the TLC (TV network) Series "Escaping the Prophet"and Lissa Golaski in "Depth", the film prequel to the Soma (video game)
by Frictional Games.

 She is also known for directing and producing short films containing messages of social influence and coming of age themes including "MISSING" and "Almost
Boyfriends".
</item 10>
=== END ITEM id=38042405ea1b45d2 ===

=== ITEM id=c18b046ae7735fe5 ===
Gold sentences and facts:
Sentence 0:  Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest commercial success is the erotic thriller "Chloe" (2009).
  Fact 0.0: Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter".
  Fact 0.1: "The Sweet Hereafter" was released in 1997.
  Fact 0.2: Egoyan received two Academy Award nominations for "The Sweet Hereafter".
  Fact 0.3: Egoyan's biggest commercial success is the erotic thriller "Chloe".
  Fact 0.4: "Chloe" was released in 2009.
Sentence 1: Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".
  Fact 1.0: Chloe is a 2009 erotic thriller film.
  Fact 1.1: Chloe was directed by Atom Egoyan.
  Fact 1.2: Chloe is a remake of the 2003 French film "Nathalie...".

Retrieved items:
<item 1>
The entries cover diverse films and careers: crime drama *Deadfall* and its *Arsenal* follow-up; *Rogue One*’s rebels stealing Death Star plans; Mel Gibson,
Nicolas Cage, James Garner and Tom Cruise filmographies; Everest disaster drama *Everest*; integration-themed sports films *Remember the Titans* and *McFarland,
USA*; acclaimed *True Grit* and *The Hurt Locker*; espionage film *The Ipcress File*; Vietnam
</item 1>

<item 2>
Tom Cruise filmography
His next role was as anti-war activist Ron Kovic in the drama adaptation of Kovic's memoir of the same name, "Born on the Fourth of July" (1989).

 For his performance Cruise received the Golden Globe Award for Best Actor – Motion Picture Drama and his first nomination for the Academy Award for Best Actor.
</item 2>

<item 3>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 3>

<item 4>
The entries highlight acclaimed Israeli cinema: *The Band’s Visit* (2007), directed by Eran Kolirin, and *Hunting Elephants* (2013), Reshef Levi’s crime comedy
starring Sasson Gabai, Moni Moshonov and Patrick Stewart. *Yana’s Friends* (1999) won 10 Ophir Awards and Karlovy Vary’s Crystal Globe. Joseph Cedar won Cannes’
2011 screenplay prize for *Footnote
</item 4>

<item 5>
Beyond the Sea (1991 film)
Beyond the Sea (translit.

 Me'ever Layam) is a 1991 Israeli drama film directed by Jacob Goldwasser.

 It stars Aryeh Moskona, Dafna Rechter, Moti Giladi.

 The film, a family drama set in the 1960s, was critically acclaimed, and won the Ophir Award for Best Film.

 The film was selected as the Israeli entry for the Best Foreign Language Film at the 64th Academy Awards, but was not accepted as a nominee.
</item 5>

<item 6>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 6>

<item 7>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 7>

<item 8>
The entries cover films across action, comedy, romance, horror, and drama. Bruce Willis stars in titles including *Last Man Standing* (a credited *Yojimbo*
remake), *Unbreakable*, *Signs*, and *Die Hard with a Vengeance*. Shyamalan’s *Signs* explores faith and alien encounters; *Unbreakable* launches a superhero
trilogy. Other highlights include *The Whole Nine Yards*, *Striking Distance*, *Mercury Rising
</item 8>

<item 9>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 9>

<item 10>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 10>
=== END ITEM id=c18b046ae7735fe5 ===
