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

=== ITEM id=d0f537a77dc4da2f ===
Gold sentences and facts:
Sentence 0: Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal Navy surface vessels and submarines.
  Fact 0.0: Rosyth Dockyard is a large naval dockyard
  Fact 0.1: Rosyth Dockyard is on the Firth of Forth
  Fact 0.2: Rosyth Dockyard is at Rosyth, Fife, Scotland
  Fact 0.3: Rosyth Dockyard is owned by Babcock Marine
  Fact 0.4: Rosyth Dockyard formerly undertook refitting of Royal Navy surface vessels
  Fact 0.5: Rosyth Dockyard formerly undertook refitting of Royal Navy submarines
Sentence 1: The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.
  Fact 1.0: The Queen Elizabeth class is a class of two aircraft carriers
  Fact 1.1: The Queen Elizabeth class aircraft carriers belong to the United Kingdom's Royal Navy

Retrieved items:
<item 1>
Goliath (Rosyth)
Goliath is a crane in Rosyth Dockyard, Scotland, with a lift capacity of 1000 t , the largest in Britain.
</item 1>

<item 2>
Queen Elizabeth-class aircraft carrier
The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.

 The first,  , was named on 4 July 2014, with her ship commissioning planned for 2017, and an initial operating capability expected in 2018.

 The second, HMS "Prince of Wales" , is scheduled to be launched around Summer 2017, followed by commissioning in 2020 and service thereafter.

 On 5 September 2014, at the NATO 2014 Wales summit, the Prime Minister David Cameron announced that the second carrier would be brought into service, ending
years of uncertainty surrounding its future.

 This was confirmed by the November 2015 Government Strategic Defence Review, with both carriers entering service, one being available at any time.
</item 2>

<item 3>
Rosyth Dockyard
Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal
Navy surface vessels and submarines.

 Before its privatisation in the 1990s it was formally the Royal Naval Dockyard Rosyth.

 Its primary role is now as integration site for the Royal Navy's newest aircraft carriers - the "Queen Elizabeth"-class.
</item 3>

<item 4>
Rosyth railway station
Rosyth railway station serves the town of Rosyth in Fife, Scotland.

 The station is managed by Abellio ScotRail and lies on the Fife Circle Line, 23.6 km north of Edinburgh Waverley .

 It was opened in 1917 by the North British Railway (as "Rosyth Halt") to serve the nearby naval dockyard.
</item 4>

<item 5>
DFDS
DFDS is Northern Europe's largest shipping and logistics company.

 The company's name is an abbreviation of Det Forenede Dampskibs-Selskab (literally "The United Steamship Company").

 DFDS was founded in 1866, when C.F. Tietgen merged the three biggest Danish steamship companies of that day.
</item 5>

<item 6>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 6>

<item 7>
HMS Prince of Wales (R09)
HMS "Prince of Wales" is the second "Queen Elizabeth"-class aircraft carrier under construction for the Royal Navy, with plans for active service from 2020.

 She is the seventh Royal Navy ship to have the name HMS "Prince of Wales".

 Construction of the ship began in 2011 at Rosyth Dockyard and in April 2016, was said to be around 80% structurally complete.
</item 7>

<item 8>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 8>

<item 9>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 9>

<item 10>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 10>
=== END ITEM id=d0f537a77dc4da2f ===

=== ITEM id=f1a8e8b29d227c89 ===
Gold sentences and facts:
Sentence 0: Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic series.
  Fact 0.0: Levni Yilmaz was born in 1973.
  Fact 0.1: Levni Yilmaz is a San Francisco based independent film maker.
  Fact 0.2: Levni Yilmaz is an artist.
  Fact 0.3: Levni Yilmaz is a publisher.
  Fact 0.4: Levni Yilmaz is best known for his 'Tales of Mere Existence' animated comic series.
Sentence 1: Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.
  Fact 1.0: Pamela B. Green is American.
  Fact 1.1: Pamela B. Green is a film producer.
  Fact 1.2: Pamela B. Green is a film director.
  Fact 1.3: Pamela B. Green is known for her work in feature film titles.
  Fact 1.4: Pamela B. Green is known for her work in motion graphics.

Retrieved items:
<item 1>
İsmet Ergün
İsmet Ergün (born 1950) is a Berlin artist and stage designer of Turkish provenance.

 Alongside her theatre work she has worked in the world of cinema as an art director and as a production designer.

 The short film which she directed, and for which she herself wrote the screenplay, was singled out for commendation at the 2007 Locarno Film Festival: it also
won the "Best German Film 2007" at the later the same year.
</item 1>

<item 2>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 2>

<item 3>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 3>

<item 4>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 4>

<item 5>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 5>

<item 6>
Pamela B. Green
Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.

 She is the director and producer of the documentary "Be Natural: The Untold Story of Alice Guy-Blache".
</item 6>

<item 7>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 7>

<item 8>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 8>

<item 9>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 9>

<item 10>
Lev Yilmaz
Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic
series.
</item 10>
=== END ITEM id=f1a8e8b29d227c89 ===

=== ITEM id=91166da47c6c56d4 ===
Gold sentences and facts:
Sentence 0: Mastodon is an American heavy metal band from Atlanta, Georgia.
  Fact 0.0: Mastodon is an American heavy metal band.
  Fact 0.1: Mastodon is from Atlanta, Georgia.
Sentence 1:  Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals in studio (except Bill Kelliher).
  Fact 1.0: Mastodon was formed in 2000.
  Fact 1.1: Mastodon is composed of bassist Troy Sanders.
  Fact 1.2: Mastodon is composed of guitarist Brent Hinds.
  Fact 1.3: Mastodon is composed of guitarist Bill Kelliher.
  Fact 1.4: Mastodon is composed of drummer Brann Dailor.
  Fact 1.5: Troy Sanders, Brent Hinds, and Brann Dailor perform vocals in the studio.
  Fact 1.6: Bill Kelliher does not perform vocals in the studio.
Sentence 2:  Mastodon has released seven studio albums, as well as a number of other records.
  Fact 2.0: Mastodon has released seven studio albums.
  Fact 2.1: Mastodon has released a number of other records.
Sentence 3:  Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.
  Fact 3.0: Mastodon's second full-length release is "Leviathan".
  Fact 3.1: "Leviathan" is a concept album.
  Fact 3.2: "Leviathan" is based on the novel "Moby-Dick".
  Fact 3.3: "Moby-Dick" was written by Herman Melville.
Sentence 4: Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric Erlandson.
  Fact 4.0: Hole was an American alternative rock band.
  Fact 4.1: Hole formed in Los Angeles, California in 1989.
  Fact 4.2: Hole was formed by singer and guitarist Courtney Love.
  Fact 4.3: Hole was formed by lead guitarist Eric Erlandson.

Retrieved items:
<item 1>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 1>

<item 2>
Hole (band)
Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric
Erlandson.

 The band had a revolving line-up of bassists and drummers, their most prolific being drummer Patty Schemel, and bassists Kristen Pfaff (d. 1994) and Melissa
Auf der Maur.
</item 2>

<item 3>
Mastodon (band)
Mastodon is an American heavy metal band from Atlanta, Georgia.

 Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals
in studio (except Bill Kelliher).

 They all perform vocals at live shows.

 Their musical style features progressive concepts and unique instrumentation.

 Mastodon has released seven studio albums, as well as a number of other records.

 The band's debut album, "Remission", released in 2002, garnered significant critical acclaim for its unique sound.

 Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.

 Three magazines awarded the record "Album of the Year" in 2004: "Revolver", "Kerrang!

" and "Terrorizer".
</item 3>

<item 4>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 4>

<item 5>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 5>

<item 6>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 6>

<item 7>
Pantera
Pantera was an American heavy metal band from Arlington, Texas.

 The group was formed in 1981 by the Abbott brothers – drummer Vinnie Paul and guitarist Dimebag Darrell – along with lead vocalist Terry Glaze.

 Bassist Rex Brown joined the band the following year, replacing Tommy D. Bradford, who was the unofficial original.

 Having started as a glam metal band, Pantera released four albums during the 1980s.

 Looking for a new and heavier sound, Pantera replaced Glaze with Phil Anselmo in late 1986 and released "Power Metal" in 1988.

 With its fifth album, 1990's "Cowboys from Hell", Pantera introduced a groove metal sound.

 Pantera's sixth album, 1992's "Vulgar Display of Power", exhibited an even heavier sound. "

Far Beyond Driven" (1994) debuted at number one on the "Billboard" 200.
</item 7>

<item 8>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 8>

<item 9>
Blackdiamondskye
Blackdiamondskye was a 2010 North American concert tour by American bands Alice in Chains, Mastodon, and Deftones.

 Blackdiamondskye is a portmanteau of the three band's most recent albums: Alice in Chains' "Black Gives Way to Blue", Deftones' "Diamond Eyes", and Mastodon's
"Crack the Skye".

 The tour kicked off on September 16 in Chicago and culminated on October 16 in Paradise, Nevada.

 The tour lasted for 19 shows over the course of one month and featured shows at both outdoor amphitheatres and indoor venues.
</item 9>

<item 10>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 10>
=== END ITEM id=91166da47c6c56d4 ===

=== ITEM id=d0b39aca77e420f9 ===
Gold sentences and facts:
Sentence 0: Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.
  Fact 0.0: Big Fish is a musical.
  Fact 0.1: Big Fish has music by Andrew Lippa.
  Fact 0.2: Big Fish has lyrics by Andrew Lippa.
  Fact 0.3: Big Fish has a book by John August.
Sentence 1:  He is a resident artist at the Ars Nova Theater in New York City.
  Fact 1.0: He is a resident artist at the Ars Nova Theater.
  Fact 1.1: The Ars Nova Theater is in New York City.

Retrieved items:
<item 1>
Marc Blitzstein
Blitzstein also composed music for films, such as "Surf and Seaweed" (1931) and "The Spanish Earth" (1937), and he contributed two songs to the original 1960
production of Hellman's play "Toys in the Attic".
</item 1>

<item 2>
Aaron Barrett
Aaron Asher Barrett (born August 30, 1974) is the lead singer, lead guitarist, and primary songwriter of the American ska-punk band Reel Big Fish.

 Prior to forming Reel Big Fish, Barrett played trombone in another ska-punk act, The Scholars, along with future Reel Big Fish bandmates Scott Klopfenstein and
Grant Barry.
</item 2>

<item 3>
Robert Lopez
Robert Lopez (born February 23, 1975) is an American songwriter of musicals, best known for co-creating "The Book of Mormon" and "Avenue Q", and for composing
the songs featured in the Disney animated film "Frozen".

 He is the youngest of only twelve people who have won an Emmy, a Grammy, an Oscar and a Tony Award, and the quickest (10 years) to win all four.
</item 3>

<item 4>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 4>

<item 5>
Big Fish (musical)
Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.

 It is based on Daniel Wallace’s 1998 novel, "", and the 2003 film "Big Fish" written by August and directed by Tim Burton.
</item 5>

<item 6>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 6>

<item 7>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 7>

<item 8>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 8>

<item 9>
Andrew Lippa
Andrew Lippa (December 22, 1964) is an American composer, lyricist, book writer, performer, and producer.

 He is a resident artist at the Ars Nova Theater in New York City.
</item 9>

<item 10>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 10>
=== END ITEM id=d0b39aca77e420f9 ===

=== ITEM id=bff0a0c757e16154 ===
Gold sentences and facts:
Sentence 0: Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.
  Fact 0.0: Innocence is a 2013 American horror drama film.
  Fact 0.1: Innocence was directed by Hilary Brougher.
  Fact 0.2: Hilary Brougher co-wrote Innocence with Tristine Skyler.
Sentence 1:  The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
  Fact 1.0: The movie Innocence stars Sophie Curtis.
  Fact 1.1: The movie Innocence stars Kelly Reilly.
  Fact 1.2: The movie Innocence stars Graham Phillips.
  Fact 1.3: The movie Innocence stars Linus Roache.
  Fact 1.4: The movie Innocence stars Sarah Sutherland.
  Fact 1.5: The movie Innocence stars Stephanie March.
Sentence 2: Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.
  Fact 2.0: Stephanie Caroline March was born on 23, 1974.
  Fact 2.1: Stephanie Caroline March is an American actress.
Sentence 3:  She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
  Fact 3.0: Stephanie March is best known for her role as Alexandra Cabot.
  Fact 3.1: The role of Alexandra Cabot was in a long-running NBC series.

Retrieved items:
<item 1>
Kim Johnston Ulrich
Kim Johnston Ulrich (born March 24, 1955) is an American actress.

 From 1983 to 1986, Ulrich played the role of Diana McColl on "As the World Turns".

 In 1988, she guest-starred in two episodes of "Werewolf".

 In 1990, Ulrich appeared on the TV series "Wings" as Carol, Brian's ex-wife.

 In 1993, Ulrich appeared in the pilot episode of "" as Dr. Antoinette Baines, a villainous scientist.

 She appeared in a third season episode of "Highlander: The Series" in 1995.

 She appeared in the first episode of season 4 of "" as a conniving wife in "Murder by Friendly Fire" in 1996.
</item 1>

<item 2>
Gloria Garayua
Gloria Garayua (born October 18, 1978 in New York City) is an American film and television actress.

 Garayua made her major film debut in the 2005 comedy film "Fun With Dick and Jane", and is now commonly cast in guest roles on long-running series such as "Six
Feet Under", "Weeds" and "The Shield".

 After being cast in an ongoing role in "Grey's Anatomy", Garayua has played recurring roles in other popular TV shows such as "Cougar Town" and "How to Get
Away with Murder".
</item 2>

<item 3>
Conviction (2006 TV series)
Conviction is an American television drama on NBC that debuted as a mid-season replacement on Friday, March 3, 2006.

 The cast includes Stephanie March reprising her "" role as Alexandra Cabot.

 In the series, Cabot returns to New York City and becomes a Bureau Chief ADA supervising a group of young but talented assistant district attorneys after a
stint in the Witness Protection Program.

 Other cast members include Eric Balfour, Anson Mount, Jordan Bridges, Julianne Nicholson, Milena Govich, and J. August Richards.
</item 3>

<item 4>
In My Sleep
In My Sleep is a 2010 suspense thriller film written, directed and produced by film director Allen Wolf.

 It stars Philip Winchester, Lacey Chabert, Tim Draxl, Abigail Spencer and Kelly Overton.

 It also features Kirsten Vangsness from Criminal Minds in a cameo role and Tony Hale from Veep in a cameo as well.

 The story is about a massage therapist with chronic insomnia who fears he may have murdered a good friend while sleepwalking.
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
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 6>

<item 7>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 7>

<item 8>
Stephanie March
Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.

 She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
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
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 10>
=== END ITEM id=bff0a0c757e16154 ===

=== ITEM id=4837b082a41dd19e ===
Gold sentences and facts:
Sentence 0: Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.
  Fact 0.0: Old Smokey is a euphemistic name for the state prison electric chair in New Jersey.
  Fact 0.1: The New Jersey state prison electric chair, called Old Smokey, is on display at the New Jersey State Police Museum.
Sentence 1:  The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
  Fact 1.0: The chair's most notorious victim was Richard Hauptmann.
  Fact 1.1: Richard Hauptmann was executed in the chair.
  Fact 1.2: Richard Hauptmann was found guilty of abducting and killing Charles Augustus Lindbergh, Jr.
  Fact 1.3: The abduction and killing of Charles Augustus Lindbergh, Jr. happened in 1932.
  Fact 1.4: Charles Augustus Lindbergh, Jr. was 20 months old.
  Fact 1.5: Charles Augustus Lindbergh, Jr. was the son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh.
  Fact 1.6: The case is referred to as the Lindbergh kidnapping (see Lindbergh kidnapping).
Sentence 2:  The Lindbergh kidnapping became known as "The Crime of the Century".
  Fact 2.0: The Lindbergh kidnapping became known as 'The Crime of the Century'.

Retrieved items:
<item 1>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 1>

<item 2>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 2>

<item 3>
Old Sparky
Old Sparky is the nickname of the electric chairs in Arkansas, Connecticut, Florida, Georgia, Illinois, Kentucky, Nebraska, New York, Ohio, Oklahoma, South
Carolina, Texas, Virginia, and West Virginia.

 Old Smokey was the nickname of the electric chairs used in New Jersey, Pennsylvania, and Tennessee.
</item 3>

<item 4>
Go! (Common song)
"Go!"

 is the third single from the Common album "Be".

 It is produced by Kanye West, who also performs backing vocals for the track alongside John Mayer.

 The track's percussion is handled by Num Amuntehu, while its scratches are provided by A-Trak.

 Its beat contains a sample from "Old Smokey" by Linda Lewis.

 Its lyrics deal with sexual fantasies.
</item 4>

<item 5>
Old Smokey
Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.

 The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus
Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
</item 5>

<item 6>
The Big House (1930 film)
The Big House is a 1930 American Pre-Code crime drama film directed by George W. Hill, released by Metro-Goldwyn-Mayer, and starring Chester Morris, Wallace
Beery, Lewis Stone and Robert Montgomery.

 The supporting cast features Leila Hyams, George F. Marion, J. C. Nugent, Karl Dane and Tom Kennedy.

 The story and dialogue were written by Frances Marion, with additional dialogue by Joe Farnham and Martin Flavin.

 The story was inspired by a spate of prison riots in 1929 and resulting federal investigation.

 In response, George Hill wrote a twenty-seven page story treatment called "The Reign of Terror: A Story of Crime and Punishment".

 Irving Thalberg gave the go ahead for the screenplay and assigned Frances Marion to work with George Hill.
</item 6>

<item 7>
Midnight Mystery
Midnight Mystery is a 1930 American mystery film directed by George B. Seitz, from a screenplay by Beulah Marie Dix, adapted from the play, "Hawk Island", by
Howard Irving Young.

 Betty Compson starred, leading an ensemble cast which included Hugh Trevor, Lowell Sherman, Rita La Roy, Ivan Lebedeff, Raymond Hatton, June Clyde and Marcelle
Corday.
</item 7>

<item 8>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 8>

<item 9>
Richard Hauptmann
Bruno Richard Hauptmann (November 26, 1899 – April 3, 1936) was a German-born carpenter who was convicted of the abduction and murder of the 20-month-old son of
aviator Charles Lindbergh and his wife Anne Morrow Lindbergh.

 The Lindbergh kidnapping became known as "The Crime of the Century".

 Hauptmann proclaimed his innocence to the end, but he was convicted of first degree murder and executed in 1936 in the electric chair at the New Jersey State
Prison.
</item 9>

<item 10>
Johnny Jack Nounes
Johnny Jack Nounes, also known as the "Beau Brummell of Galveston", was an organized crime boss in Galveston, Texas, United States, during the early 1900s.

 He, with one-armed George Musey, led the Downtown Gang, one of the two gangs which controlled most of the Galveston Crime Syndicate until the early 1930s.

 They fought for control of the island against the rival Beach Gang led by Ollie Quinn and Dutch Voight.

 As the prohibition era began, his gang came to be one of the dominant forces in the Galveston Crime Syndicate.

 Galveston became the main port of entry for liquor supply in Texas and many parts of the Midwest.
</item 10>
=== END ITEM id=4837b082a41dd19e ===

=== ITEM id=b230bb0e9fd1b3ed ===
Gold sentences and facts:
Sentence 0:  He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.
  Fact 0.0: He played college football at Hampton University.
  Fact 0.1: He attended Riverdale High School.
  Fact 0.2: Riverdale High School is in Riverdale, Georgia.
Sentence 1: Riverdale is a city in Clayton County, Georgia, United States.
  Fact 1.0: Riverdale is a city.
  Fact 1.1: Riverdale is in Clayton County, Georgia.
  Fact 1.2: Clayton County, Georgia is in the United States.

Retrieved items:
<item 1>
Saint Michael the Archangel Church (Monroe, MI)
In 1918 the parish built the present building of St. Michael School which is now a part of Monroe Catholic Elementary Schools.

 The movement to establish the parish started in 1845 by 14 German immigrants who wanted to worship in their native language.

 The only nearby church was St. Mary of the Immaculate Conception right across the river where St. Michael stands today.

 St. Mary's only worshiped in French and wouldn't hold services in other languages like German and Irish.

 The German and Irish families were forced to move to the second floor of the rectory to have mass in their native languages.

 When the German families were able to acquire land they were eager to leave St. Mary's to start a parish of their own.

 They first purchased an empty lot Humphrey Street but the plan was soon abandoned.

 They then purchased the first mayor of Monroe's estate to build the church and it remains the current site of the church.
</item 1>

<item 2>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 2>

<item 3>
Saint Michael the Archangel Church (Monroe, MI)
Saint Michael the Archangel Church (St. Michael Church) is located on the west side of the city of Monroe, Michigan along the River Raisin.

 It is home to 1,200 families and it is one of most important religious institutes in Monroe County.

 It was founded in the year 1852.

 The present building was built from 1866-1867.

 It is in the Archdiocese of Detroit.

 Its current priest is Rev. Phillip Ching.

 When the parish was first established, the mayor of Monroe palatial residence was remodeled as a temporary church.

 It was used as a church on the first floor and on the second it was used for the school.

 Later it would only be used as St. Michael School.

 Then in 1866 the cornerstone for the present church was laid.

 The large 187 foot steeple wasn't added until 1883.

 In 1874 the 3-story rectory was built east of the church.

 In 1918 the parish built the present building of St.
</item 3>

<item 4>
North Boone Community Unit School District 200
The current principal is Matt Klett.

 All students who graduate from one of the district elementary schools will consolidate at the fifth grade in one school, called North Boone Upper Elementary
School, whose principal is Mike Greenlee;all students move on to North Boone Middle School, headed by Jeremiah Auble, which educates those in grades seven and
eight.

 The last leg of education this district can provide is to those from grade nine to grade twelve; the facility is called North Boone High School, whose
principal is named Jacob Hubert.

 The current superintendent is Steven Baule.

 The mascot of the district high school is the Viking.
</item 4>

<item 5>
Richard Harris (American football)
Richard Drew Harris (January 21, 1948 – July 26, 2011) was an American football defensive end who played seven seasons in the National Football League.

 He was and All-American in 1970 for Grambling and was drafted in the first round (5th overall pick) of the 1971 NFL Draft by the Philadelphia Eagles.

 He was named All-Rookie in 1971, playing defensive end.

 Harris spent seven seasons as a defensive end in the NFL, the first three with the Philadelphia Eagles, the next two with the Chicago Bears and the final two
with the Seattle Seahawks.
</item 5>

<item 6>
Lake Worth Community High School
Lake Worth Community High School is a B State of Florida rated public high school located in Lake Worth, Florida.

 Established in 1922 as Lake Worth High School, it is currently one of Palm Beach County's largest schools.

 It has six magnet programs: the award-winning Air Force JROTC led by former Multi-National Award-winning Drill Teams, Tyranny and Fidelis; Medicine and Allied
Health; Child Care and Development; Criminal Justice; Culinary and Drafting and Design.

 The Palm Beach County School Board added the word Community to the names of all public high schools, including Lake Worth High School, in the 1980s.
</item 6>

<item 7>
Michael Ola
Michael Ola (born April 19, 1988) is an American football offensive tackle for the Buffalo Bills of the National Football League (NFL).

 He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.

 He has also been a member of the Jacksonville Sharks, Montreal Alouettes, Miami Dolphins, Chicago Bears, San Diego Chargers, Detroit Lions, Seattle Seahawks
and New York Giants.
</item 7>

<item 8>
Rich Central High School
Rich Central High School or RCHS is a public four-year high school located in Olympia Fields, Illinois, a southern suburb of Chicago, Illinois, in the United
States.

 The Rich Central Campus serves the cities of Matteson, Richton Park, Country Club Hills, Chicago Heights, Tinley Park, and parts of Olympia Fields.

 It is a part of Rich Township District 227, which also includes Rich East High School and Rich South High School.

 Although the school is located in Olympia Fields, it does not serve the entire village.

 Some Olympia Fields students attend high school at Homewood-Flossmoor High School and Bloom High School, depending on the subdivision where the student
resides.
</item 8>

<item 9>
Two Rivers High School (Arkansas)
Two Rivers High School is a public high school for students in grades 7 through 12 located in Ola, Arkansas, United States.

 Two Rivers High School is administered by the Two Rivers School District.

 The Two Rivers High School and school district serves 650 sqmi of rural communities including Ola, Plainview, Rover, and Casa.

 Two Rivers High School is one of four high schools within Yell County and the sole high school administered by the Two Rivers School District.
</item 9>

<item 10>
Riverdale, Georgia
Riverdale is a city in Clayton County, Georgia, United States.

 The population was 15,134 at the 2010 census, up from 12,478 in 2000.

 Riverdale is a suburb just south of Hartsfield–Jackson Atlanta International Airport and is part of the Atlanta metropolitan area.
</item 10>
=== END ITEM id=b230bb0e9fd1b3ed ===

=== ITEM id=a263c6574df9b0f5 ===
Gold sentences and facts:
Sentence 0:  This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's "Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
  Fact 0.0: This episode is a milestone in the history of the series.
  Fact 0.1: In this episode, Snowball II is killed off.
  Fact 0.2: Homer's mother died in season 19's Mona Leaves-a.
  Fact 0.3: Excluding the death of Homer's mother, the killing of Snowball II is to date the closest thing to an actual Simpson family member being killed off.
Sentence 1: Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".
  Fact 1.0: Mona Penelope Simpson's maiden name is Olsen.
  Fact 1.1: Mona Penelope Simpson is a recurring fictional character.
  Fact 1.2: Mona Penelope Simpson is a character in the animated television series The Simpsons.

Retrieved items:
<item 1>
“Moho House,” episode 21 of *The Simpsons*’ 28th season and the series’ 617th episode, aired May 7, 2017. The material also surveys the show’s seasons,
episodes, characters, shorts, theme music, and comics. Featured stories include Moe’s stolen Flaming Homer recipe, Mona Simpson’s appearances and death, the
family’s whale rescue, and episodes centered on Lisa, Bart, Homer, and their Springfield neighbors.
</item 1>

<item 2>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 2>

<item 3>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 3>

<item 4>
Girly Edition
"Girly Edition" is the twenty-first episode in the ninth season of the American animated television series "The Simpsons".

 It originally aired on the Fox network in the United States on April 19, 1998.

 In the episode, Lisa and Bart Simpson must co-anchor a new news program, though when Bart is seen as a more successful news anchor, Lisa becomes jealous and
seeks revenge.

 Meanwhile, in the subplot, Homer Simpson gets a monkey helper because of his laziness.

 "Girly Edition" was the first episode written by Larry Doyle and was directed by Mark Kirkland.

 Much of the subplot was inspired by the film "Monkey Shines".

 Critics gave the episode positive reviews and it was well received by Lisa's voice actress Yeardley Smith.
</item 4>

<item 5>
My Mother the Carjacker
"My Mother the Carjacker" is the second episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It first aired on the Fox network in the United States on November 9, 2003.

 Homer receives a cryptic message in the newspaper informing him to come to a certain place at midnight, and soon discovers that the person who wrote the
message is his mother, Mona Simpson.

 It was written by Michael Price and directed by Nancy Kruse.

 Glenn Close makes her second of six guest spots as Homer's mother.

 It has a direct link from the season seven episode, "Mother Simpson".

 It was nominated for a Writers Guild of America Award in 2004.

 In its original run, the episode received 12.4 million viewers.
</item 5>

<item 6>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 6>

<item 7>
Lisa the Drama Queen
"Lisa the Drama Queen" is the ninth episode of the twentieth season of "The Simpsons".

 It originally aired on the Fox network in the United States on January 25, 2009 and guest starred Emily Blunt as Juliet.

 A special version of the end credits theme was performed by Fall Out Boy, although they do not guest star in the actual episode.

 This was the last episode in the series to be presented only in standard definition, the first regular episode to begin right after the opening credits without
a commercial break, with an episode now having four segments, and the last episode to use the original opening sequence starting from Season 2.

 Brian Kelley returns as a writer after five seasons of being absent from the show.

 The episode is very loosely based on the 1994 film "Heavenly Creatures".
</item 7>

<item 8>
I, (Annoyed Grunt)-bot
"I, (Annoyed Grunt)-Bot", also known as ""I, D'oh-Bot"", is the ninth episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It originally aired on the Fox network in the United States on January 11, 2004.

 This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's
"Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
</item 8>

<item 9>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 9>

<item 10>
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 10>
=== END ITEM id=a263c6574df9b0f5 ===

=== ITEM id=08a6bda450c5b722 ===
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
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 1>

<item 2>
Lyman: first U.S. boys’ reform school (Massachusetts, c.1886–1971), now historic. Massachusetts International Academy: Chinese-student U.S.-college prep
boarding school. Polytech’Paris-UPMC: French research-linked engineering school in an 11-school network. Cal Poly Pomona: public polytechnic; library holds
670,580 books; engineering enrolls 5,600+; environmental design 1,600+; women’s basketball compet
</item 2>

<item 3>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 3>

<item 4>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 4>

<item 5>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 5>

<item 6>
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
</item 6>

<item 7>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 7>

<item 8>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 8>

<item 9>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=08a6bda450c5b722 ===

=== ITEM id=13cbd2ace46f6a7a ===
Gold sentences and facts:
Sentence 0: Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.
  Fact 0.0: Elizabeth Austin was born in 1958.
  Fact 0.1: Elizabeth Austin is an American writer.
  Fact 0.2: Elizabeth Austin lives in Oak Park, Illinois.
Sentence 1:  She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".
  Fact 1.0: Elizabeth Austin has also written for The Washington Monthly.
  Fact 1.1: Elizabeth Austin has also written for Ladies Home Journal.
  Fact 1.2: Elizabeth Austin has also written for Good Housekeeping.
  Fact 1.3: Elizabeth Austin has also written for Self.
  Fact 1.4: Elizabeth Austin has also written for Shape.
Sentence 2: Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight fitness) and became the number one women's fitness magazine.
  Fact 2.0: Shape is a monthly English language fitness magazine.
  Fact 2.1: Shape was started by Weider Publications in 1981.
  Fact 2.2: Shape was founded by Christine MacIntyre.
  Fact 2.3: Christine MacIntyre is a pioneer in women's free weight fitness.
  Fact 2.4: Shape became the number one women's fitness magazine.

Retrieved items:
<item 1>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 1>

<item 2>
Muscle &amp; Fitness (British magazine)
Muscle & Fitness is a British fitness magazine, which contains articles on strength and fitness training, diet plans and strategies for men and women, and tips
and advice.
</item 2>

<item 3>
Ingrid Romero
Ingrid Romero (born July 10, 1985) is an international fitness model, fitness competitor, and fitness coach.

 Training & Fitness Magazine called her "one of the most recognized faces in the fitness industry," and "one of the most accomplished female fitness stars
ever."
</item 3>

<item 4>
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
</item 4>

<item 5>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 5>

<item 6>
Anais Zanotti
Anais Zanotti (born January 8, 1986 in Saint-Tropez) is a French model and stunt woman of French and Italian ancestry who lives in Miami, Florida.

 She has been featured in magazines such as "Playboy", "FHM", "Maxim", "GQ", "Cosmopolitan", "Vogue", and "People", Men's Fitness magazine, Muscle and Fitness,
Women Fitness magazine and more.

 She has over 40 magazine covers.

 Zanotti is also a certified skydiver with over 1250 jumps.GS .
</item 6>

<item 7>
Wendy Austin
Dr Wendy Elizabeth Austin Hewitt, Wendy Austin is a BBC journalist and broadcaster in BBC Northern Ireland, and former presenter of BBC Radio Ulster’s current
affairs programme "Talkback".
</item 7>

<item 8>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 8>

<item 9>
Elizabeth Austin (writer)
Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.

 Austin has lived in the Midwestern United States all her life.

 She began her journalism career at the now-defunct City News Bureau of Chicago and has written articles for national news magazines such as "Newsweek" and
"Time", in addition to a wide range of other national and regional publications, such as "People", and the "Chicago Tribune".

 She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".

 She has often written about medicine and health care.

 Austin also has been heard on WBEZ, the National Public Radio station in Chicago, as well as the BBC in the UK and Scotland.

 She has appeared on the Today Show and on ITV in Europe.
</item 9>

<item 10>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 10>
=== END ITEM id=13cbd2ace46f6a7a ===

=== ITEM id=527be623c535c4e8 ===
Gold sentences and facts:
Sentence 0: "Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".
  Fact 0.0: 'Collateral Damage' is the eleventh episode of the third season of the crime-thriller television series 'Millennium'.
  Fact 0.1: 'Millennium' is a crime-thriller television series.
  Fact 0.2: 'Collateral Damage' is in the third season of 'Millennium'.
Sentence 1:  "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.
  Fact 1.0: 'Collateral Damage' featured a guest appearance by O'Quinn.
  Fact 1.1: 'Collateral Damage' featured a guest appearance by Jacinda Barrett.
  Fact 1.2: 'Collateral Damage' featured a guest appearance by James Marsters.
  Fact 1.3: 'Collateral Damage' featured a guest appearance by radio host Art Bell.
  Fact 1.4: Art Bell appeared as himself in 'Collateral Damage'.
Sentence 2:  Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
  Fact 2.0: Art Bell's radio show 'Coast to Coast AM' was among Perry's inspirations for the script.
Sentence 3: Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.
  Fact 3.0: Coast to Coast AM is an American late-night radio talk show.
  Fact 3.1: Coast to Coast AM deals with a variety of topics.
Sentence 4:  The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
  Fact 4.0: Coast to Coast AM now airs seven nights a week.
  Fact 4.1: Coast to Coast AM airs from 1:00 a.m. to 5:00 a.m. Eastern Time Zone.

Retrieved items:
<item 1>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 1>

<item 2>
List of Jimmy Fallon games and sketches
The following is a list of recurring games, sketches, and other comedy routines from the NBC late night talk show "The Tonight Show Starring Jimmy Fallon", and
its predecessor, "Late Night with Jimmy Fallon".

 The sketches feature host Jimmy Fallon, house band The Roots, announcer/sidekick Steve Higgins, the show's writers, celebrity guests, and audience members.

 Most of the skits below appeared only on "Late Night", while some have carried over to the "Tonight Show".
</item 2>

<item 3>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 3>

<item 4>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 4>

<item 5>
CNN World News
CNN World News, a program that airs on CNN International and CNN International Asia Pacific.

 It is supplemented by CNN World News Asia and CNN World News Europe The show's traditional time run is 24-hours if it is followed by CNN World News Middle East
The show's regular presenters include Errol Barnett.

 Its main role is to update viewers of the latest news in the world.

 It contains a weather update from the CNN World Weather Forecast News.

 CNN World News can usually air up to three times on weekends, and is known to be bringing the latest on a story.
</item 5>

<item 6>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 6>

<item 7>
Collateral Damage (Millennium)
"Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".

 It premiered on the Fox network on January 22, 1999.

 The episode was written by Michael R. Perry, and directed by Thomas J. Wright.

 "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.

 Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
</item 7>

<item 8>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 8>

<item 9>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 9>

<item 10>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 10>
=== END ITEM id=527be623c535c4e8 ===
