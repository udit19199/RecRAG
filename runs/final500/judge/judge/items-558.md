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

=== ITEM id=4895f45c9a5e1540 ===
Gold sentences and facts:
Sentence 0: The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.
  Fact 0.0: The Night of the Arts is an annual event.
  Fact 0.1: The Night of the Arts is known in Finnish as 'Taiteiden yö'.
  Fact 0.2: The Night of the Arts is known in Swedish as 'Konstens natt'.
  Fact 0.3: The Night of the Arts is held in several major cities in Finland.
  Fact 0.4: The Night of the Arts is held in late August.
Sentence 1:  It is one of many White Night festivals held worldwide.
  Fact 1.0: The Night of the Arts is one of many White Night festivals.
  Fact 1.1: White Night festivals are held worldwide.
Sentence 2: The White Nights are a kind of all-night arts festival held in many cities in the summer.
  Fact 2.0: The White Nights are a kind of all-night arts festival.
  Fact 2.1: The White Nights festival is held in many cities.
  Fact 2.2: The White Nights festival is held in the summer.
Sentence 3:  The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are early and darkness is never complete.
  Fact 3.0: 'White nights' is the name given in areas of high latitude to the weeks around the summer solstice in June.
  Fact 3.1: During white nights, sunsets are late.
  Fact 3.2: During white nights, sunrises are early.
  Fact 3.3: During white nights, darkness is never complete.

Retrieved items:
<item 1>
Night of the Arts
The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.

 It is one of many White Night festivals held worldwide.
</item 1>

<item 2>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 2>

<item 3>
Green Man Festival
Green Man is an independent music and arts festival held annually in mid-August in the Brecon Beacons, Wales since 2003.

 It has evolved into a 20,000 capacity 4-day event, showcasing predominantly live music (in particular alternative, indie, rock, folk, dance and americana),
with additional events showcasing literature, film, comedy, theatre and poetry.

 2015 saw 1,500 multi-arts acts perform across 17 stages.

 The festival site is divided into 10 areas, each offering a unique festival experience.

 Ceilidhs, all-night bonfires and secret gigs all add to the festival's unique identity.
</item 3>

<item 4>
Satumaa
"Satumaa" (roughly, in English "The Fabled Land" or "The Fairytale Land") is the quintessential Finnish tango.

 It was written by Unto Mononen, and published in 1955.

 The most famous recording is probably the one made by Reijo Taipale in 1962.

 The lyrics tell a story of a distant land beyond the sea — a happy paradise — however, the narrator can only reach it in his thoughts.

 The song has been recorded countless times, mainly by male Finnish tango singers.
</item 4>

<item 5>
Artscape (festival)
Artscape is an annual art festival held in the Mount Royal neighborhood of Baltimore, Maryland in July.

 Since its first annual event in 1982, it has become the largest free arts festival in America.

 It has boasted acts such as Ray Charles, Aretha Franklin, and Matisyahu in the past, attracting over 350,000 people from the city, and surrounding areas.

 Film programming during Artscape is provided by Maryland Film Festival.

 There are artists in a variety of visual and performing media.

 Events are free and open to the public.
</item 5>

<item 6>
White Night festivals
The White Nights are a kind of all-night arts festival held in many cities in the summer.

 The original festival is the White Nights Festival held in Saint Petersburg, Russia.

 The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are
early and darkness is never complete.

 In Saint Petersburg, the Sun does not set until after 10 p.m., and the twilight lasts almost all night.
</item 6>

<item 7>
Great Northern Arts Festival
The Great Northern Arts Festival is held each year in Inuvik, Northwest Territories.

 Lasting approximately ten days, it is an annual event held each summer.

 It was established in 1989 by Sharlene Alexander and Sue Rose.

 While the festival's mandate is to serve Northern artists and craftspeople, artists from around the world also attend.

 The festival is designed as a venue for artists to see each other's work and to experiment with new applications.

 It includes arts of all types, such as fiber art and Music of the Northwest Territories.

 Artists partake in demonstrations, masterclasses, seminars, and workshops.

 The festival's gallery offers pieces for sale.

 There are cultural presentations each evening which include concerts, dance, fashion, music, and storytelling.

 Media coverage brings festival highlights to viewing audiences.
</item 7>

<item 8>
Graph query row:
connected_entities: ['Night of the Arts', 'White Night festivals']
relationships: ['PART_OF']
</item 8>

<item 9>
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
</item 9>

<item 10>
Tanglewood Music and Arts Festival
Tanglewood Music and Arts Festival is an annual two-day music, comedy and arts event held within the grounds of Narrow Water Castle, County Down, Northern
Ireland.

 The festival, founded in 2011, has attracted significant attention both within the local area and around Northern Ireland with buses coming from as far as
Belfast to the event.

 For the 2011 and 2012 festivals Tanglewood was held on one day; however for 2013 it has expanded into a two-day camping festival.
</item 10>
=== END ITEM id=4895f45c9a5e1540 ===

=== ITEM id=6cc1ddc93d9aa33a ===
Gold sentences and facts:
Sentence 0: "Grow Some Funk of Your Own" is a song by Elton John.
  Fact 0.0: 'Grow Some Funk of Your Own' is a song by Elton John.
Sentence 1: Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.
  Fact 1.0: Sir Elton Hercules John was born Reginald Kenneth Dwight.
  Fact 1.1: Sir Elton Hercules John was born on 25 March 1947.
  Fact 1.2: Sir Elton Hercules John is an English singer.
  Fact 1.3: Sir Elton Hercules John is an English pianist.
  Fact 1.4: Sir Elton Hercules John is an English composer.
Sentence 2:  John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.
  Fact 2.0: Elton John owned Watford Football Club from 1976 to 1987.
  Fact 2.1: Elton John owned Watford Football Club from 1997 to 2002.

Retrieved items:
<item 1>
Afrodiziak
Afrodiziak was a British singing group composed of Caron Wheeler, Claudia Fontaine, and later Naomi Thompson, that was active in the 1980s.

 As a duo, Wheeler and Fontaine were best known for performing backing vocals on The Jam's final single "Beat Surrender" in 1982 (with whom they also performed
on their final tour) and Elvis Costello's 1983 album "Punch the Clock", especially its single, the international hit "Everyday I Write the Book".

 After the addition of a third member, Naomi Thompson, they went on to sing for acts such as Heaven 17 and Howard Jones.

 Afrodiziak performed backing vocals on the hit single Free Nelson Mandela, including the a cappella intro, staging it at The Tube (Channel4) in March 1984.

 Heaven 17's 1984 album, "How Men Are", featured them prominently, especially on the singles "Sunset Now" and "And That's No Lie".
</item 1>

<item 2>
Blitz Kids
The Blitz Kids were a group of young people who frequented the weekly Blitz club-night in Covent Garden, London in 1979-80, and are credited with launching the
New Romantic subcultural movement.

 It was Andrew Czezowski (Ex manager of The Damned) and Susan Carrington who went on to start The Roxy club with Barry Jones that introduced Steve Strange and
Rusty Egan to the Blitz in Covent Garden, which started the New Romantic Movement.

 Among the many attendees were Rusty Egan, Steve Strange, Boy George, Marilyn, Alice Temple, Perri Lister, Princess Julia, Philip Sallon, Carl Teper and Martin
Degville (later to be the frontman of Tony James' Sigue Sigue Sputnik).

 Crucially, the club lay between two art colleges (St Martin's School and Central School) and became a testbed for student fashion designers who set London
ablaze during the 1980s.
</item 2>

<item 3>
Grow Some Funk of Your Own
"Grow Some Funk of Your Own" is a song by Elton John.

 It was released as a single in 1976 and comes from his album "Rock of the Westies".

 It shared its A-side status with "I Feel Like a Bullet (In the Gun of Robert Ford)".

 The song is considered to be one of John's heavier rock songs.

 The song went to #14 on the "Billboard" Hot 100, but in Britain broke a five-year run of successful singles by failing to reach the Top 50 despite extensive
radio play.

 Although the singles chart listed only "Funk" and not "Bullet", the latter song was given equal billing in later "Billboard" singles chart books.
</item 3>

<item 4>
The Happenings
The Happenings are a pop music group that originated in the 1960s.

 Members of the original group, created in the spring of 1961 and initially called "The Four Grads" because all had just graduated from high school in Paterson,
New Jersey, were Bob Miranda, David Libert, Tom Giuliano, and Ralph DiVito.

 In 1968 DiVito was replaced by Bernie LaPorta and Lenny Conforti also joined to play drums in the touring band.

 Both LaPorta and Conforti took a hiatus from the northern New Jersey band, The Emerald Experience, to play and tour with The Happenings.

 That lineup performed mostly at colleges and universities until 1970, when Libert left the band to manage other groups, including George Clinton and
Parliament-Funkadelic, Living Colour, Brian Auger, Vanilla Fudge, The Runaways (Cherie Currie, Joan Jett, Lita Ford), Mother's Finest, Alice Cooper and Evelyn
"Champagne" King.
</item 4>

<item 5>
Firm Biz
"Firm Biz" is the first single by the Hip hop supergroup The Firm from their singular collaborative LP "The Album".

 The song was produced by L.E.S, who based the song's track on a sample of Teena Marie's 1981 hit "Square Biz".

 "Firm Biz" also features a chorus performed by former En Vogue singer Dawn Robinson, reworked from the "Square Biz" chorus.

 AZ raps the first verse of the song, Nas the second, and Foxy Brown the third and final verse.

 The song never reached any US chart positions, but was a UK Top 20 hit in 1997.

 A remix of the song was released, featuring a verse from Half-A-Mill and a chorus by Mary J. Blige.

 The word "feds" is censored in the explicit version, in verse 2, performed by Nas, when he says "Never that though Black .4-4's for feds".
</item 5>

<item 6>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 6>

<item 7>
Elton John
He has also composed music, produced records, and has occasionally acted in films.

 John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.

 He is an honorary Life President of the club, and in 2014 had a stand named after him at the club's home stadium.
</item 7>

<item 8>
Knitting Factory Records
Knitting Factory Records is an independent American music label that is notable for promoting a variety of artists, including the music of deceased Nigerian
political activist Fela Kuti.

 The label promotes a variety of music artists including Ages and Ages, Ash Black Bufflo, Cuong Vu, Graham Haynes, Femi Kuti, Gary Lucas, Lumerians, Thomas
Chapin, Patrolled By Radar, Joe Morris, Rachid Taha, Seun Kuti, and Shilpa Ray and her Happy Hookers.
</item 8>

<item 9>
South Liverpool F.C.
South Liverpool Football Club is a football club based in Liverpool, England, founded as a phoenix club of a club of the same name.

 They are currently members of the West Cheshire League Division One and play at Jericho Lane in Otterspool.

 The club's colours are white shirts, black shorts and red socks.
</item 9>

<item 10>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 10>
=== END ITEM id=6cc1ddc93d9aa33a ===

=== ITEM id=0cf75ab96cff42dc ===
Gold sentences and facts:
Sentence 0: "White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".
  Fact 0.0: 'White People Renovating Houses' is the first episode of the twenty-first season of the American animated television series South Park.
Sentence 1:  It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
  Fact 1.0: 'White People Renovating Houses' is the 278th episode of South Park overall.
  Fact 1.1: 'White People Renovating Houses' first aired on Comedy Central in the United States.
  Fact 1.2: 'White People Renovating Houses' first aired on September 13, 2017.
Sentence 2: The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
  Fact 2.0: The twenty-first season of South Park is an American animated sitcom.
  Fact 2.1: The twenty-first season of South Park premiered on Comedy Central on September 13, 2017.
  Fact 2.2: The twenty-first season of South Park will contain ten episodes.

Retrieved items:
<item 1>
Something Wall-Mart This Way Comes
"Something Wall-Mart This Way Comes" is the ninth episode of the eighth season of the American animated television series "South Park", and the 120th episode of
the series overall.

 It originally aired on Comedy Central in the United States on November 3, 2004.

 In the episode, a Wall-Mart is built in South Park, and the people start to get addicted to shopping from it, due to its irresistibly attractive bargains, thus
leading many businesses in South Park to close down.

 The four boys have to fight against Wall-Mart and to find a way to stop it from taking over the entire town.
</item 1>

<item 2>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 2>

<item 3>
White People Renovating Houses
"White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".

 It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
</item 3>

<item 4>
Put It Down (South Park)
"Put It Down" is the second episode in the twenty-first season of the American animated television series "South Park".

 The 279th episode of the series overall, it first aired on Comedy Central in the United States on September 20, 2017.
</item 4>

<item 5>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 5>

<item 6>
Chickenlover
"Chickenlover" is the fourth episode in the second season of the American animated television series "South Park".

 The 16th episode of the series overall, it originally aired on Comedy Central in the United States on May 27, 1998.

 The episode was written by series co-creators Trey Parker and Matt Stone, along with David R. Goodman, and directed by Parker.

 In the episode, Officer Barbrady resigns as South Park's only police officer because of his illiteracy.

 Anarchy ensues, just as chickens are mysteriously being molested across South Park.

 Barbrady enlists the help of the boys to learn to read and discover who is molesting the chickens.

 Cartman, meanwhile, masquerades as a police officer.
</item 6>

<item 7>
South Park (season 21)
The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
</item 7>

<item 8>
South Park (season 10)
The tenth season of "South Park", an American animated television series created by Trey Parker and Matt Stone, began airing on March 22, 2006.

 The tenth season concluded after 14 episodes on November 15, 2006.

 This is the first season in which Kenny does not die and the last season featuring Isaac Hayes (the voice of Chef) as Hayes quit the show following the
backlash behind season nine's "Trapped in the Closet" episode.

 This season also had a minor controversy when the Halloween episode "Hell on Earth 2006" depicted "The Crocodile Hunter's" Steve Irwin with a stingray lodged
in his chest getting thrown out of Satan's Halloween party for not being in costume.

 Episode 2 in this season is the last one with the Braniff Airlines logo.

 All the episodes in this season were written and directed by Trey Parker.
</item 8>

<item 9>
Holiday Special (South Park)
"Holiday Special" is the third episode in the twenty-first season of the American animated television series "South Park".

 The 280th episode of the series overall, it first aired on Comedy Central in the United States on September 27, 2017.
</item 9>

<item 10>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 10>
=== END ITEM id=0cf75ab96cff42dc ===

=== ITEM id=361eb2e5df97a43f ===
Gold sentences and facts:
Sentence 0: Sir Ridley Scott (born 30 November 1937) is an English film director and producer.
  Fact 0.0: Sir Ridley Scott was born on 30 November 1937.
  Fact 0.1: Sir Ridley Scott is English.
  Fact 0.2: Sir Ridley Scott is a film director.
  Fact 0.3: Sir Ridley Scott is a film producer.
Sentence 1:  Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
  Fact 1.0: Ridley Scott had a commercial breakthrough with the science-fiction horror film Alien, released in 1979.
  Fact 1.1: Ridley Scott's best known works include the neo-noir dystopian science fiction film Blade Runner, released in 1982.
  Fact 1.2: Ridley Scott's best known works include the historical drama Gladiator, released in 2000.
  Fact 1.3: Gladiator won the Best Picture Oscar.
  Fact 1.4: Ridley Scott's best known works include the science fiction film The Martian, released in 2015.
Sentence 2:  A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work behind the camera, with Griffith and Joseph Henabery as his mentors.
  Fact 2.0: Ridley Scott was a collaborator of D.W. Griffith.
  Fact 2.1: Ridley Scott appeared in The Birth of a Nation, released in 1915.
  Fact 2.2: Ridley Scott appeared in Intolerance, released in 1916.
  Fact 2.3: Ridley Scott gave up acting in 1917.
  Fact 2.4: Ridley Scott gave up acting to concentrate on work behind the camera.
  Fact 2.5: Griffith was a mentor of Ridley Scott.
  Fact 2.6: Joseph Henabery was a mentor of Ridley Scott.

Retrieved items:
<item 1>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 1>

<item 2>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 2>

<item 3>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 3>

<item 4>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 4>

<item 5>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 5>

<item 6>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 8>

<item 9>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 9>

<item 10>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 10>
=== END ITEM id=361eb2e5df97a43f ===

=== ITEM id=dd250f5af93ae7be ===
Gold sentences and facts:
Sentence 0: The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.
  Fact 0.0: The presidency of Ronald Reagan began on January 20, 1981
  Fact 0.1: Ronald Reagan was inaugurated as President of the United States on January 20, 1981
  Fact 0.2: The presidency of Ronald Reagan ended on January 20, 1989
Sentence 1: George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice President of the United States from 1981 to 1989.
  Fact 1.0: George Herbert Walker Bush was born June 12, 1924
  Fact 1.1: George Herbert Walker Bush is an American politician
  Fact 1.2: George Herbert Walker Bush was the 41st President of the United States from 1989 to 1993
  Fact 1.3: George Herbert Walker Bush was the 43rd Vice President of the United States from 1981 to 1989

Retrieved items:
<item 1>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 1>

<item 2>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 2>

<item 3>
Elliott Abrams
Elliott Abrams (born January 24, 1948) is a former American diplomat, lawyer and political scientist who served in foreign policy positions for Presidents
Ronald Reagan and George W. Bush.

 Abrams was convicted of withholding information from Congress about the Iran–Contra affair while serving under Reagan, but was pardoned by President George H.
W. Bush.
</item 3>

<item 4>
Reagan Democrat
A Reagan Democrat is a traditionally Democratic voter in the United States, referring especially to white working-class Rust Belt residents, who defected from
their party to support Republican President Ronald Reagan in either or both of the 1980 and 1984 elections as well as Republican Presidents George H. W. Bush in
the 1988 election and George W. Bush in either or both of the 2000 and 2004 elections.

 Part of this group also defected to Donald Trump, the Republican candidate in 2016, who won in Pennsylvania, Ohio, Michigan, Wisconsin, and Iowa, all states
that voted for Reagan but went Democratic in 2008 and 2012.
</item 4>

<item 5>
Ronald Reagan judicial appointment controversies
During President Ronald Reagan's presidency, he nominated at least twelve people for various federal appellate judgeship who were not confirmed.

 In some cases, the nominations were not processed by the Democratic-controlled Senate Judiciary Committee before Reagan's presidency ended, while in other
cases, nominees were rejected by the Senate Judiciary Committee or even blocked by unfriendly members of the Republican Party.

 Three of the nominees were renominated by Reagan's successor, President George H. W. Bush.

 Two of the nominees, Ferdinand Francis Fernandez and Guy G. Hurlbutt, were nominated after July 1, 1988, the traditional start date of the unofficial Thurmond
Rule during a presidential election year.

 Eight of the twelve seats eventually were filled by appointees of President George H. W. Bush.
</item 5>

<item 6>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 6>

<item 7>
Presidency of Ronald Reagan
The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.

 Reagan, a Republican, took office as the 40th United States president following a landslide win over Democratic incumbent President Jimmy Carter in the 1980
presidential election.

 Reagan was succeeded by his vice president, George H. W. Bush, who won the 1988 presidential election with Reagan's support.
</item 7>

<item 8>
First inauguration of Ronald Reagan
The first inauguration of Ronald Reagan as the 40th President of the United States was held on Tuesday, January 20, 1981, on the West Front of the United States
Capitol in Washington, D.C..

 This was the first inauguration to be held on the building's west side.

 The inauguration marked the commencement of the first four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.

 Chief Justice Warren E. Burger administered the presidential oath of office to Reagan, who placed his hand upon a family Bible given to him by his mother, open
to .

 Associate Justice Potter Stewart administered the vice presidential oath to Bush.
</item 8>

<item 9>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 9>

<item 10>
Second inauguration of Ronald Reagan
The second inauguration of Ronald Reagan as President of the United States was held privately on Sunday, January 20, 1985 in the Grand Foyer of the White House,
and publicly the following day, January 21, 1985 at the United States Capitol in Washington, D.C. The inauguration marked the commencement of the second
four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.
</item 10>
=== END ITEM id=dd250f5af93ae7be ===

=== ITEM id=cd0b9865917ee2ca ===
Gold sentences and facts:
Sentence 0: Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".
  Fact 0.0: Stephanocereus is a genus of cactus.
  Fact 0.1: Stephanocereus is from Brazil.
  Fact 0.2: Stephanocereus is related to "Arrojadoa".
Sentence 1:  They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).
  Fact 1.0: The plants in this sentence are mostly liana-forming.
  Fact 1.1: The plants in this sentence are found in the Old World tropics.
  Fact 1.2: The plants in this sentence range from South China, Indonesia, Malaysia and the Philippines.
  Fact 1.3: The plants in this sentence range westward to Madagascar, with two species there.
  Fact 1.4: The plants in this sentence range westward to the Seychelles, with one species there.
  Fact 1.5: The plants in this sentence range southward to Australia, with three species there.
  Fact 1.6: The plants in this sentence range southward to New Caledonia, with one species there.
  Fact 1.7: The plants in this sentence range northward to India, with one species there.
  Fact 1.8: The plants in this sentence range northward to Sri Lanka, with one species there.

Retrieved items:
<item 1>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 1>

<item 2>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 2>

<item 3>
Nepenthes veitchii
Nepenthes veitchii ( or ; after James Veitch, nurseryman of the Veitch Nurseries), or Veitch's Pitcher-Plant, is a "Nepenthes" species from the island of
Borneo.

 The plant is widespread in north-western Borneo and can also be found in parts of Kalimantan.

 "Nepenthes veitchii" usually grows as an epiphyte, though the form from Bario seems to be strictly terrestrial and has not been observed to climb trees.
</item 3>

<item 4>
Robert Cantley
Robert Cantley is a conservationist and Managing Director of Borneo Exotics, a Sri Lankan-based plant nursery specialising in tissue-cultured and seed-grown
"Nepenthes" species and hybrids.

 Cantley has contributed to a number of papers on "Nepenthes".

 The species "Nepenthes robcantleyi", the natural hybrid "Nepenthes × cantleyi" and the cultivar "Nepenthes" 'Cantley's Red' are named after him.
</item 4>

<item 5>
Nepenthes tenax
Nepenthes tenax ( ; from Latin: "tenax" "tenacious") is a lowland species of tropical pitcher plant native to northern Queensland, Australia.

 It is the third "Nepenthes" species recorded from the continent and its second endemic species.

 "Nepenthes tenax" is closely related to the two other Australian "Nepenthes" species: "N. mirabilis" and "N. rowaniae".
</item 5>

<item 6>
Man-eating tree
Man-eating tree can refer to any of the various legendary carnivorous plants large enough to kill and consume a person or other large animal.

 The carnivorous plant with the largest known traps is probably Nepenthes rajah, which produces pitchers up to 41 cm tall with a volume up to 3.5 l .

 The pitcher of this species are designed to trap arthropods.

 However, the same bait may also attract rodents like the summit rat ("Rattus baluensis") and the Mountain treeshrew ("Tupaia montana").

 Only very rarely will the rodents fall into the large pitchers of this species.

 Other large carnivorous plants that have similar properties include Nepenthes robcantleyi and Nepenthes attenboroughi.
</item 6>

<item 7>
Nepenthes chaniana
Nepenthes chaniana ( ; after Datuk Chan Chew Lun, Managing Director of Natural History Publications (Borneo)) is a tropical pitcher plant species belonging to
the genus "Nepenthes".

 It is characterised by a dense indumentum of long, white hairs.

 Pitchers are cylindrical and mostly white to yellow in colouration.

 "Nepenthes chaniana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. boschiana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 7>

<item 8>
Nepenthes × kinabaluensis
Nepenthes × kinabaluensis , or the Kinabalu Pitcher-Plant, is the natural hybrid between "N. rajah" and "N. villosa".

 It was first collected near Kambarangoh on Mount Kinabalu, Borneo by Lilian Gibbs in 1910 and later mentioned by John Muirhead Macfarlane as ""Nepenthes" sp."

 in 1914.

 Although Macfarlane did not formally name the plant, he noted that "[a]ll available morphological details suggest that this is a hybrid between "N. villosa"
and "N. rajah"".

 It was finally described in 1976 by Shigeo Kurata as "N. × kinabaluensis".

 The name was first published in "Nepenthes of Mount Kinabalu", but was a "nomen nudum" at the time as it lacked an adequate description and information on the
type specimen.

 The name was subsequently published validly by Kurata in 1984.
</item 8>

<item 9>
Nepenthes boschiana
Nepenthes boschiana ( ; after Johannes van den Bosch), or Bosch's Pitcher-Plant, is a tropical pitcher plant endemic to Borneo.

 It is most closely allied to "N. faizaliana".

 "Nepenthes borneensis" is considered a synonym of this species.

 "Nepenthes boschiana" has no known natural hybrids.

 No valid forms or varieties have been described.

 "Nepenthes boschiana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. chaniana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 9>

<item 10>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 10>
=== END ITEM id=cd0b9865917ee2ca ===

=== ITEM id=a57ff8745f31c25c ===
Gold sentences and facts:
Sentence 0: Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to 2003.
  Fact 0.0: Václav Havel was born on 5 October 1936.
  Fact 0.1: Václav Havel died on 18 December 2011.
  Fact 0.2: Václav Havel was a Czech writer.
  Fact 0.3: Václav Havel was a former dissident.
  Fact 0.4: Václav Havel served as the 1st President of the Czech Republic.
  Fact 0.5: Václav Havel served as President of the Czech Republic from 1993 to 2003.
Sentence 1: Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.
  Fact 1.0: Penelope Fitzgerald was born on 17 December 1916.
  Fact 1.1: Penelope Fitzgerald died on 28 April 2000.
  Fact 1.2: Penelope Fitzgerald was English.
  Fact 1.3: Penelope Fitzgerald was a Booker Prize-winning novelist.
  Fact 1.4: Penelope Fitzgerald was a poet.
  Fact 1.5: Penelope Fitzgerald was an essayist.
  Fact 1.6: Penelope Fitzgerald was a biographer.

Retrieved items:
<item 1>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 1>

<item 2>
Letters to Olga
Letters to Olga ("Czech:Dopisy Olze") is a book of compiled letters written by Czech playwright, dissident, and future president, Václav Havel to his wife Olga
Havlová during his nearly four-year imprisonment from May 1979 to March 1983.

 (Havel was released when he came down with a high fever and received a medical discharge).

 Havel was imprisoned by the communist regime of then Czechoslovakia for being one of the leaders of The Committee for the Defense of the Unjustly Prosecuted
(VONS) - most of whom had been signatories of the human rights document Charter 77.
</item 2>

<item 3>
Václav Havel Tribute Concert
Tribute to Václav Havel (Czech: Pocta Václavu Havlovi) was an event held in memory of Václav Havel, the last Czechoslovak and the first Czech President, writer,
playwright and human rights activist.

 The concert took place in Lucerna Music Bar in Prague on 23 December 2011.

 Václav Havel died on Sunday 18 December 2011 at 10:15 a.m.
</item 3>

<item 4>
Graph query row:
person: Václav Havel
birth_date: 5 October 1936
</item 4>

<item 5>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 5>

<item 6>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 6>

<item 7>
Václav Havel Human Rights Prize
The Václav Havel Human Rights Prize is an annual €60,000 award established in 2013 by the Parliamentary Assembly of the Council of Europe, the Václav Havel
Library and the Charta 77 Foundation, to honour "outstanding" civil society action in defence of human rights, in Europe and beyond.
</item 7>

<item 8>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 8>

<item 9>
Czech presidential election, 1993
The 1993 Czech presidential election was held on 26 January 1993 to elect the first President of independent Czech Republic.

 The first election was done indirectly by the members of the Czech Republic Parliament.

 Václav Havel was elected President.

 The election was complicated only by attacks of Republicans against Havel and by bomb threat to the Parliament.
</item 9>

<item 10>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 10>
=== END ITEM id=a57ff8745f31c25c ===

=== ITEM id=271e777a5f4d9d7b ===
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
No. 82 Wing RAAF
No. 82 Wing is the strike and reconnaissance wing of the Royal Australian Air Force (RAAF).

 It is headquartered at RAAF Base Amberley, Queensland.

 Coming under the control of Air Combat Group, the wing operates F/A-18F Super Hornet multirole fighters and Pilatus PC-9 forward air control aircraft.

 Its units include Nos. 1 and 6 Squadrons, operating the Super Hornet, and No. 4 Squadron, operating the PC-9.
</item 1>

<item 2>
Boeing EA-18G Growler
The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.

 The EA-18G replaced the Northrop Grumman EA-6B Prowlers in service with the United States Navy.

 The Growler's electronic warfare capability is primarily provided by Northrop Grumman.

 The EA-18G began production in 2007 and entered operational service with the US Navy in late 2009.

 Australia has also purchased twelve EA-18Gs, which entered service with the Royal Australian Air Force in 2017.
</item 2>

<item 3>
Boeing F/A-18E/F Super Hornet
The Boeing F/A-18E and F/A-18F Super Hornet are twin-engine carrier-capable multirole fighter aircraft variants based on the McDonnell Douglas F/A-18 Hornet.

 The F/A-18E single-seat and F/A-18F tandem-seat variants are larger and more advanced derivatives of the F/A-18C and D Hornet.

 The Super Hornet has an internal 20 mm M61 rotary cannon and can carry air-to-air missiles and air-to-surface weapons.

 Additional fuel can be carried in up to five external fuel tanks and the aircraft can be configured as an airborne tanker by adding an external air refueling
system.
</item 3>

<item 4>
Type 002 aircraft carrier
The Type 002 aircraft carrier is a second generation Chinese aircraft carrier design.

 The ship will be the first Chinese aircraft carrier to be equipped with a catapult.

 According to the latest report, the construction of this carrier has been rescheduled for a decision on the catapults, where the latest competition results
suggested that steam catapults will be used in the Type 002 aircraft carrier.

 The addition of aircraft catapult will give China the ability to launch various fixed wing support aircraft like airborne early warning (AEW) and carrier
onboard delivery aircraft.

 Fighters can also be launched at full combat load and reach its full potential unlike current Type 001 aircraft carriers operated by Chinese navy.
</item 4>

<item 5>
VFA-2
Strike Fighter Squadron 2 (VFA-2) also known as the "Bounty Hunters" is a United States Navy F/A-18F Super Hornet strike fighter squadron based at Naval Air
Station Lemoore, California.

 Their tail code is NE and their callsign is "Bullet".

 They are attached to Carrier Air Wing 2 (CVW-2), a composite unit made up of a wide array of aircraft performing a variety of combat and support missions that
deploy aboard the "Carl Vinson" .
</item 5>

<item 6>
VAQ-209
Electronic Attack Squadron 209 (VAQ-209) is a United States Navy Reserve electronic attack squadron.

 Known as the "Star Warriors", the squadron flies the EA-18G Growler carrier-based electronic warfare jet aircraft.

 Based at Naval Air Station Whidbey Island, WA, it is assigned to the Tactical Support Wing.
</item 6>

<item 7>
No. 1 Squadron RAAF
No. 1 Squadron is a Royal Australian Air Force (RAAF) squadron headquartered at RAAF Base Amberley, Queensland.

 Controlled by No. 82 Wing, it is equipped with Boeing F/A-18F Super Hornet multi-role fighters.

 The squadron was formed under the Australian Flying Corps in 1916 and saw action in the Sinai and Palestine Campaigns during World War I.

 It flew obsolete Royal Aircraft Factory B.E.2s, B.E.12s, Martinsyde G.100s and G.102s, as well as Airco DH.6s, Bristol Scouts and Nieuport 17s, before
re-equipping with the R.E.8 in October 1917 and finally the Bristol Fighter in December.

 Its commanding officer in 1917–18 was Major Richard Williams, later known as the "Father of the RAAF".

 Disbanded in 1919, No. 1 Squadron was re-formed on paper as part of the RAAF in 1922, and re-established as an operational unit three years later.
</item 7>

<item 8>
VAQ-138
Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at
Naval Air Station Whidbey Island, Washington.

 Their radio callsign is "RAMPAGE".
</item 8>

<item 9>
VAQ-135
Electronic Attack Squadron 135 (VAQ-135), known as the "Black Ravens", is a United States Navy electronic attack squadron that currently operates the EA-18G
Growler carrier-based electronic warfare jet aircraft.

 The squadron is permanently stationed at Naval Air Station Whidbey Island with a radio callsign of ""Thunder"".
</item 9>

<item 10>
VFA-154
Strike Fighter Squadron 154 (VFA-154), also known as the "Black Knights", is a United States Navy strike fighter squadron stationed at Naval Air Station
Lemoore.

 The Black Knights are an operational fleet squadron flying the F/A-18F Super Hornet.

 They are currently attached to Carrier Air Wing Eleven and deployed aboard the aircraft carrier USS "Nimitz" .

 Their tailcode is NH and their callsign is "Knight".
</item 10>
=== END ITEM id=271e777a5f4d9d7b ===

=== ITEM id=c471b5468440331e ===
Gold sentences and facts:
Sentence 0: John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
  Fact 0.0: John André lived from 2 May 1750 to 2 October 1780.
  Fact 0.1: John André was a British Army officer.
  Fact 0.2: John André was hanged as a spy by the Continental Army during the American Revolutionary War.
  Fact 0.3: John André was hanged for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York, to the British.
Sentence 1: Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American Continental Army, and later defected to the British Army.
  Fact 1.0: Benedict Arnold lived from January 14, 1741 (Old Style: January 3, 1740) to June 14, 1801.
  Fact 1.1: Benedict Arnold was a general during the American Revolutionary War.
  Fact 1.2: Benedict Arnold fought for the American Continental Army.
  Fact 1.3: Benedict Arnold later defected to the British Army.

Retrieved items:
<item 1>
Siege of Yorktown
The Siege of Yorktown, also known as the Battle of Yorktown, the Surrender at Yorktown, German Battle or the Siege of Little York, ending on October 19, 1781,
at Yorktown, Virginia, was a decisive victory by a combined force of American Continental Army troops led by General George Washington and French Army troops
led by the Comte de Rochambeau over a British Army commanded by British peer and Lieutenant General Charles Cornwallis.

 The culmination of the Yorktown campaign, the siege proved to be the last major land battle of the American Revolutionary War in the North American theater, as
the surrender by Cornwallis, and the capture of both him and his army, prompted the British government to negotiate an end to the conflict.

 The battle boosted faltering American morale and revived French enthusiasm for the war, as well as undermining popular support for the conflict in Great
Britain.
</item 1>

<item 2>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
</item 2>

<item 3>
List of Continental Army units
The Continental Army was the national army of first the Thirteen Colonies, and then the independent United States, during the American Revolutionary War,
established by a resolution of the Congress on June 14, 1775, three days before the Battle of Bunker Hill, where it saw its first action under that title.

.

 The Continental Congress took a number of steps in the spring of 1775 to create the army in response to the Battles of Lexington and Concord in April and the
seizure of Fort Ticonderoga in May.

 The units composing the Continental Army changed frequently, especially in the first two years of the war.

 From 1777 to the close of the war, the organization of the Continental Army became progressively more systematic and sophisticated.

 The Continental Army that served at Yorktown in 1781 bore very little resemblance to the Continental Army that blockaded Boston in 1775.
</item 3>

<item 4>
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 4>

<item 5>
John André
John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting
Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
</item 5>

<item 6>
Benjamin Lincoln
Benjamin Lincoln (January 24, 1733 (O.S. January 13, 1732) – May 9, 1810) was an American army officer.

 He served as a major general in the Continental Army during the American Revolutionary War.

 Lincoln is notable for being involved in three major surrenders during the war: his participation in the Battles of Saratoga (sustaining a wound shortly
afterward) contributed to John Burgoyne's surrender of a British army, he oversaw the largest American surrender of the war at the 1780 Siege of Charleston,
and, as George Washington's second in command, he formally accepted the British surrender at Yorktown.
</item 6>

<item 7>
Benjamin Logan
Benjamin Logan (c.1742 – December 11, 1802) was an American pioneer, soldier, and politician from Shelby County, Kentucky.

 As colonel of the Kentucky County, Virginia militia during the American Revolutionary War, he was second-in-command of all the militia in Kentucky.

 He was also a leader in Kentucky's efforts to become a state.

 His brother, John Logan, was the first state treasurer of Kentucky.
</item 7>

<item 8>
Pulaski's Legion
Pulaski's Legion was raised on March 28, 1778 at Baltimore, Maryland under the command of Kazimierz Pułaski for service with the Continental Army during the
American Revolutionary War.

 The Legion consisted of one troop of lancers, two troops of dragoons and 200 light infantry soldiers.

 It was one of the few cavalry regiments in the American Continental Army.
</item 8>

<item 9>
1st Virginia State Regiment
The 1st Virginia State Regiment was a regiment of Virginia Militia that fought during the American Revolutionary War.

 The regiment was authorized by the General Assembly of the Commonwealth of Virginia in December 1776 as a force of regular troops for the Commonwealth's
defense.

 In 1777, Virginia had difficulty meeting its quota for the regular line of the Continental Army.

 As a result, in July 1777 under the command of Colonel George Gibson, the regiment began a march North to temporarily join the Continental Army in the
Philadelphia Campaign.

 In January 1778, the Virginia General Assembly passed an act directing that the 1st Virginia State Regiment "now in Continental service, be continued in said
service instead of the Ninth Virginia Regiment, made prisoners by the enemy in the Battle of Germantown."

 The regiment camped at Valley Forge in the winter of 1777-78 and at Middlebrook in the winter of 1778-79 and participated in the Battle of Monmouth.

 The regiment remained in the service of the Continental Army until late 1779 when redeployed Virginia.
</item 9>

<item 10>
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 10>
=== END ITEM id=c471b5468440331e ===

=== ITEM id=6b87b15eec60565b ===
Gold sentences and facts:
Sentence 0: Park Sung-woong (born January 9, 1973) is a South Korean actor.
  Fact 0.0: Park Sung-woong was born on January 9, 1973.
  Fact 0.1: Park Sung-woong is a South Korean actor.
Sentence 1:  Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a serial killer in "The Deal" (2015).
  Fact 1.0: Park Sung-woong made his acting debut in 'No. 3' in 1997.
  Fact 1.1: Park Sung-woong has starred in several movies and television series.
  Fact 1.2: Park Sung-woong notably starred as a gangster in 'New World' (2013).
  Fact 1.3: Park Sung-woong notably starred as a serial killer in 'The Deal' (2015).
Sentence 2: The Deal (; lit.
  Fact 2.0: The Deal's literal translation is given in the next sentence, after the 'lit.' marker.
Sentence 3:  "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
  Fact 3.0: The literal translation of The Deal is 'Murder Request'.
  Fact 3.1: The Deal is a 2015 South Korean crime thriller film.
  Fact 3.2: The Deal was directed by Son Yong-ho.

Retrieved items:
<item 1>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 1>

<item 2>
A Violent Prosecutor
A Violent Prosecutor is a 2016 South Korean crime film directed by Lee Il-hyung, produced by Guk Su Ran and starring Hwang Jung-min, Kang Dong-won, Lee Sung-min
and Park Sung-woong.

 It was released in South Korea on February 3, 2016 by Showbox.
</item 2>

<item 3>
The Swindlers (2017 film)
The Swindlers is an upcoming South Korean crime drama film directed by Jang Chang-won.

 The film stars Hyun Bin, Yoo Ji-tae, Bae Seong-woo, Park Sung-woong, Nana and Ahn Se-ha.
</item 3>

<item 4>
The Boys Who Cried Wolf
The Boys Who Cried Wolf (), also known as The Shepherd, is a 2015 South Korean crime thriller drama film.

 Written and directed by Kim Jin-hwang in his first feature-length for his Korean Academy of Film Arts (KAFA)'s final year undergraduate film project, it
depicts how a former stage actor becomes involved in a murder case.
</item 4>

<item 5>
Deep Trap
Deep Trap () (previously known as Exchange) is a 2015 South Korean crime thriller film starring Ma Dong-seok, Jo Han-sun and Kim Min-kyung, and directed by Kwon
Hyung-jin.

 Based on an actual SNS crime, it depicts the terror experienced by a married couple while on a trip to an isolated island.

 It won Best Film Award in the Orient Express section at the Fantasporto in 2016.
</item 5>

<item 6>
A Company Man
A Company Man () is a 2012 South Korean action thriller film starring So Ji-sub, Lee Mi-yeon, Kwak Do-won and Kim Dong-jun.

 It is about a hitman who find himself targeted by his ex-employers after he falls in love with a single mother and quits his job.
</item 6>

<item 7>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 7>

<item 8>
Park Sung-woong
Park Sung-woong (born January 9, 1973) is a South Korean actor.

 Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a
serial killer in "The Deal" (2015).
</item 8>

<item 9>
Office (2015 South Korean film)
Office (오피스 ) is 2015 South Korean slasher-thriller film directed by Hong Won-chan and starring Go Ah-sung and Park Sung-woong.

 It is about a detective trying to figure out why a mild-mannered man has killed his family and is targeting his co-workers.

 The film premiered at the Midnight Screenings section of the 2015 Cannes Film Festival.
</item 9>

<item 10>
Fatal Intuition
Fatal Intuition (; lit.

 "It's Him") is a 2015 South Korean crime thriller film written and directed by Yun Jun-hyeong.
</item 10>
=== END ITEM id=6b87b15eec60565b ===

=== ITEM id=288720ab4bbf3912 ===
Gold sentences and facts:
Sentence 0: August or Auguste Brancart (21 July 1851 - 1894?)
  Fact 0.0: August Brancart is also known as Auguste Brancart.
  Fact 0.1: August Brancart was born on 21 July 1851.
  Fact 0.2: The end date given for August Brancart is 1894, with a question mark.
Sentence 1:  He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
  Fact 1.0: Brancart published work of the Decadent movement.
  Fact 1.1: Brancart published "Monsieur Vénus" by Rachilde.
  Fact 1.2: "Monsieur Vénus" by Rachilde is an example of Decadent movement work.
Sentence 2: The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and artificiality.
  Fact 2.0: The Decadent Movement was an artistic and literary movement.
  Fact 2.1: The Decadent Movement was a late 19th-century movement.
  Fact 2.2: The Decadent Movement was centered in Western Europe.
  Fact 2.3: The Decadent Movement followed an aesthetic ideology of excess.
  Fact 2.4: The Decadent Movement followed an aesthetic ideology of artificiality.

Retrieved items:
<item 1>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 1>

<item 2>
Hermann Brunn
Karl Hermann Brunn (1 August 1862 – 20 September 1939) was a German mathematician, known for his work in convex geometry (see Brunn–Minkowski inequality) and in
knot theory.

 Brunnian links are named after him, as his 1892 article "Über Verkettung" included examples of such links.
</item 2>

<item 3>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 3>

<item 4>
Decadent movement
The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and
artificiality.

 The visual artist Félicien Rops's body of work and Joris-Karl Huysmans's novel "Against Nature" (1884) are considered the prime examples of the decadent
movement.

 It first flourished in France and then spread throughout Europe and to the United States.

 The movement was characterized by self-disgust, sickness at the world, general skepticism, delight in perversion and employment of crude humor and a belief in
the superiority of human creativity over logic and the natural world.
</item 4>

<item 5>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 5>

<item 6>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 6>

<item 7>
Ebba Witt-Brattström
Ebba Witt-Brattström (born 1953) is a Swedish scholar in comparative literature.

 She is Professor of Literature and head of department at Södertörn University outside Stockholm, and a well-known feminist.

 In the 1970s she was a member of the feminist organisation Grupp 8, and in 2005 she was one of the founders of the feminist political organisation and party
Feministiskt Initiativ, although she later distanced herself from the organisation and criticized its alleged strong left-wing tendencies.
</item 7>

<item 8>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 8>

<item 9>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 9>

<item 10>
The Exception and the Rule
The Exception and the Rule (in German Die Ausnahme und die Regel) is a short play by German playwright Bertolt Brecht and is one of several "Lehrstücke"
(Teaching plays) he wrote around 1929/30.

 The objective of Brecht's Lehrstücke was that they be taken on tour and performed in schools or in factories to educate the masses about socialist politics.
</item 10>
=== END ITEM id=288720ab4bbf3912 ===
