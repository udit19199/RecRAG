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

=== ITEM id=89597783526096fa ===
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
Black Country Rangers F.C.
Black Country Rangers Football Club is a football club based in Rowley Regis in the West Midlands.

 They are currently members of the West Midlands (Regional) League Premier Division and play at the Beeches in Cradley, groundsharing with Cradley Town.
</item 1>

<item 2>
Liverpool F.C. league record by opponent
Liverpool Football Club is an English association football club based in Liverpool, Merseyside, which competes in the top tier of English football, for the
2015–16 season.

 The club was formed in 1892 following a disagreement between the board of Everton and club president John Houlding, who owned the club's ground, Anfield.

 The dispute over rent resulted in Everton leaving Anfield for Goodison Park, which left Houlding with an empty stadium.

 Not content for his ground to lay idle, he created his own club: Liverpool.

 Liverpool joined the Lancashire League on their foundation before the 1892–93 season.

 They ended their inaugural season as league champions, and were elected to The Football League soon afterwards.

 The club remained in The Football League until 1992, when its First Division was superseded as English football's top level by the newly formed Premier League.
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
Knitting Factory Records
Knitting Factory Records is an independent American music label that is notable for promoting a variety of artists, including the music of deceased Nigerian
political activist Fela Kuti.

 The label promotes a variety of music artists including Ages and Ages, Ash Black Bufflo, Cuong Vu, Graham Haynes, Femi Kuti, Gary Lucas, Lumerians, Thomas
Chapin, Patrolled By Radar, Joe Morris, Rachid Taha, Seun Kuti, and Shilpa Ray and her Happy Hookers.
</item 4>

<item 5>
Billy Bibby &amp; The Wry Smiles
Billy Bibby & The Wry Smiles are a British indie rock band formed in the autumn of 2015 in the area around Llandudno, North Wales and Chester, UK.

 The band is fronted by Billy Bibby, former lead guitarist and founding member of Catfish and the Bottlemen.

 Bibby parted ways with Catfish and the Bottlemen in the summer of 2014, shortly after co-writing and recording their Platinum selling debut album, "The
Balcony".

 Soon thereafter, Billy Bibby began writing songs and playing acoustic shows in various UK cities in order to test public response to his new music.

 In the spring of 2015, Billy Bibby began seeking musicians to form a new band.

 Auditions were conducted with the help of Simon Jones, formerly of The Verve.

 Band members Rob Jones (lead guitar), Matt Bassy (bass), and Michael Pearce (drums) were selected to form the four-piece ensemble.
</item 5>

<item 6>
Fayez Sarofim
Fayez Sarofim (Arabic: فايز صاروفيم‎ ‎ ) (born 1929) is a Coptic American heir to the Sarofim family fortune, fund manager for a number of Dreyfus family stock
funds, an original and second largest shareholder of Kinder Morgan (NYSE: KMI) and part owner of the NFL team Houston Texans; ranked 5th Most Valuable NFL team
worth $1.85 billion.

 He is divorced with five children, and lives in Houston, Texas.

 With an estimated current net worth of $1.91 billion, Sarofim is ranked by "Forbes" as the 847-richest person in the world, in 2015.

 His investment firms oversees over $30 billion in assets.
</item 6>

<item 7>
South Liverpool F.C.
South Liverpool Football Club is a football club based in Liverpool, England, founded as a phoenix club of a club of the same name.

 They are currently members of the West Cheshire League Division One and play at Jericho Lane in Otterspool.

 The club's colours are white shirts, black shorts and red socks.
</item 7>

<item 8>
Gordon Burns (footballer)
Gordon Burns (born 2 December 1978) is a Scottish former professional footballer who is currently manager of Troon in the Scottish Junior Football Association,
West Region.

 He has previously played in the Scottish Championship for Ayr United.
</item 8>

<item 9>
Elton John
He has also composed music, produced records, and has occasionally acted in films.

 John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.

 He is an honorary Life President of the club, and in 2014 had a stand named after him at the club's home stadium.
</item 9>

<item 10>
Ejikeme Ikwunze
Ejikeme Ikwunze, popularly called Mr Football, is an international figure in Nigeria's sporting community.

 He is a sports columnist and contributes articles to several international sports magazines as well as Nigerian Newspapers including the "National Ambassador",
"Vanguard", "ThisDay" and the "Daily Champion".

 He also features as a sports commentator on several stations such as "Federal Radio Corporation of Nigeria", the "BCA Radio", "Abia Television" and "Pacesetter
Radio".
</item 10>
=== END ITEM id=89597783526096fa ===

=== ITEM id=cb600ade4183cd29 ===
Gold sentences and facts:
Sentence 0: The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista Distribution.
  Fact 0.0: The Rescuers is a 1977 film.
  Fact 0.1: The Rescuers is an American animated film.
  Fact 0.2: The Rescuers is a comedy-drama adventure film.
  Fact 0.3: The Rescuers was produced by Walt Disney Productions.
  Fact 0.4: The Rescuers was first released on June 22, 1977.
  Fact 0.5: The Rescuers was first released by Buena Vista Distribution.
Sentence 1: The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.
  Fact 1.0: The Muppets is a 2011 film.
  Fact 1.1: The Muppets is an American film.
  Fact 1.2: The Muppets is a musical comedy film.
  Fact 1.3: The Muppets is the seventh theatrical film featuring the Muppets.

Retrieved items:
<item 1>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 1>

<item 2>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 2>

<item 3>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
</item 3>

<item 4>
Eva Gabor
Eva Gabor ( ; February 11, 1919 – July 4, 1995) was a Hungarian-born American actress, comedian, singer and socialite.

 She was widely known for her role on the 1965–71 television sitcom "Green Acres" as Lisa Douglas, the wife of Eddie Albert's character, Oliver Wendell Douglas.

 She voiced "Duchess" in the 1970 Disney film "The Aristocats", and Miss Bianca in Disney’s "The Rescuers" and "The Rescuers Down Under".

 Gabor was successful as an actress in film, on Broadway and on television.

 She was also a successful businessperson, marketing wigs, clothing and beauty products.

 Her elder sisters, Zsa Zsa and Magda Gabor, were also actresses and socialites.
</item 4>

<item 5>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 5>

<item 6>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 6>

<item 7>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 7>

<item 8>
The Muppets at Walt Disney World
The Muppets at Walt Disney World is a television special starring Jim Henson's Muppets at Walt Disney World in Lake Buena Vista, Florida.

 The special aired on NBC as part of "The Magical World of Disney" on May 6, 1990, and turned out to be the last Muppets special that Henson would work on (he
would die 10 days following its airing, on May 16).
</item 8>

<item 9>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 9>

<item 10>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 10>
=== END ITEM id=cb600ade4183cd29 ===

=== ITEM id=73ef4a36e42eb1dd ===
Gold sentences and facts:
Sentence 0: William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story writer.
  Fact 0.0: William Somerset Maugham was born 25 January 1874.
  Fact 0.1: William Somerset Maugham died 16 December 1965.
  Fact 0.2: William Somerset Maugham is better known as W. Somerset Maugham.
  Fact 0.3: William Somerset Maugham was British.
  Fact 0.4: William Somerset Maugham was a playwright.
  Fact 0.5: William Somerset Maugham was a novelist.
  Fact 0.6: William Somerset Maugham was a short story writer.
  Fact 0.7: William Somerset Maugham's name includes the post-nominal CH.
Sentence 1:  He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
  Fact 1.0: W. Somerset Maugham was among the most popular writers of his era.
  Fact 1.1: W. Somerset Maugham was reputedly the highest-paid author during the 1930s.
Sentence 2: Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.
  Fact 2.0: Thomas Merton, O.C.S.O., was born January 31, 1915.
  Fact 2.1: Thomas Merton died December 10, 1968.
  Fact 2.2: Thomas Merton was an American Catholic writer.
  Fact 2.3: Thomas Merton was a theologian.
  Fact 2.4: Thomas Merton was a mystic.
Sentence 3:  In 1949, he was ordained to the priesthood and given the name Father Louis.
  Fact 3.0: In 1949, Thomas Merton was ordained to the priesthood.
  Fact 3.1: Thomas Merton was given the name Father Louis.

Retrieved items:
<item 1>
Trio (film)
Trio (also known as W. Somerset Maugham's Trio) is a 1950 British anthology film based on three short stories by W. Somerset Maugham: "The Verger", "Mr.
Know-All" and "Sanatorium".

 Ken Annakin directed "The Verger" and "Mr. Know-All", while Harold French was responsible for "Sanatorium".
</item 1>

<item 2>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 2>

<item 3>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 3>

<item 4>
W. Somerset Maugham
William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story
writer.

 He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
</item 4>

<item 5>
Somerset Maugham Award
The Somerset Maugham Award is a British literary prize given each May by the Society of Authors.

 It is awarded to the best writer or writers of a book published in the past year who is under the age of 35.

 The prize was instituted in 1947 by William Somerset Maugham and thus bears his name.

 The award is to be spent on foreign travel.

 The total fund for each year is £12,000.
</item 5>

<item 6>
Thomas Merton
Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.

 A Trappist monk of the Abbey of Gethsemani, Kentucky, he was a poet, social activist, and student of comparative religion.

 In 1949, he was ordained to the priesthood and given the name Father Louis.
</item 6>

<item 7>
Paul Schulte
Paul Schulte OMI (1896 – 1975), was a German priest and missionary, known as the "Flying Priest", who founded MIVA ("Missionary International Vehicular
Association") to provide automobiles, boats and airplanes for the service of missions throughout the world.
</item 7>

<item 8>
List of works about Thomas Merton
The following is a list of works about Thomas Merton, publications about Thomas Merton, the Trappist monk from Abbey of Gethsemani.

 The bibliography is organized into categories.

 A separate list of works by Thomas Merton is also available.
</item 8>

<item 9>
Michael Yorke
Michael Leslie Yorke was an Anglican priest in the last decades of the 20th century and the first years of the 21st.

 He was born on 25 March 1939 and educated at Midhurst Grammar School and Magdalene College, Cambridge.

 Ordained in 1965 his first post was a curacy at Croydon Parish Church after which he served as Succentor, Precentor and Chaplain at Chelmsford Cathedral.

 Following this he was Rector of Hadstock, a Canon Residentiary at Chelmsford Cathedral, Vicar of St Margaret’s with St Nicholas, King’s Lynn and Provost of
Portsmouth Cathedral.

 In 1999 he became Dean of Lichfield, and is now in retirement as Dean Emeritus.
</item 9>

<item 10>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 10>
=== END ITEM id=73ef4a36e42eb1dd ===

=== ITEM id=5826af0e73fa17cc ===
Gold sentences and facts:
Sentence 0: Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.
  Fact 0.0: Sir Nicholas Robert Hytner was born on 7 May 1956.
  Fact 0.1: Sir Nicholas Robert Hytner is an English theatre director.
  Fact 0.2: Sir Nicholas Robert Hytner is an English film director.
  Fact 0.3: Sir Nicholas Robert Hytner is an English film producer.
Sentence 1: Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
  Fact 1.0: Maya Deren was born on April 29, 1917.
  Fact 1.1: Maya Deren died on October 13, 1961.
  Fact 1.2: Maya Deren was born Eleanora Derenkowskaia.
  Fact 1.3: The birth name Eleanora Derenkowskaia is given in Russian as Элеоно́ра Деренко́вская.
  Fact 1.4: Maya Deren was a Russian-American filmmaker.
  Fact 1.5: Maya Deren was one of the most important American experimental filmmakers in the 1940s and 1950s.
  Fact 1.6: Maya Deren was one of the most important American entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
Sentence 2:  Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
  Fact 2.0: Maya Deren was also a choreographer.
  Fact 2.1: Maya Deren was also a dancer.
  Fact 2.2: Maya Deren was also a film theorist.
  Fact 2.3: Maya Deren was also a poet.
  Fact 2.4: Maya Deren was also a lecturer.
  Fact 2.5: Maya Deren was also a writer.
  Fact 2.6: Maya Deren was also a photographer.

Retrieved items:
<item 1>
Nicholas Mele
Nicholas Mele is an actor who has starred in many movies and on television.

 His first movie role was in the 1976 movie "The Ritz".

 Other movie roles include "Some Kind of Hero" (1982) and "Young Doctors in Love" (1982).

 His most well-known film role was in the 1988 hit horror movie "" as Dennis Johnson.

 A year later, he reprised his role in the hit sequel "".

 His most recent movie is the 2003 film "The Great Gabble".
</item 1>

<item 2>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 2>

<item 3>
Mary Sweeney
Mary Sweeney is an American film producer, director, writer and film editor, who collaborated for 20 years with her husband, avant-garde American film director,
David Lynch.

 Sweeney worked with Lynch on several films and television series, most notably the original "Twin Peaks" series (1990), "Lost Highway" (1997), "The Straight
Story", (1999) and "Mulholland Drive" (2001).

 Sweeney is the Dino and Martha De Laurentiis Endowed Professor in the Writing Division of the School of Cinematic Arts at the University of Southern
California.

 She is the Chair of the Film Independent Board of Directors.
</item 3>

<item 4>
Teiji Ito
Teiji Ito (伊藤貞司 , Itō Teiji , January 22, 1935 – August 16, 1982) was a Japanese composer and performer.

 He is best known for his scores for the avant-garde films by Maya Deren.

 The independent filmmaker, Barbara Hammer, continues to promote and educate the world about both Teiji and his wife, Maya Deren.
</item 4>

<item 5>
Maya Deren
Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the
most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.

 Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
</item 5>

<item 6>
Filmworks X: In the Mirror of Maya Deren
Filmworks X: In the Mirror of Maya Deren features a score for film by John Zorn.

 The album was released on Zorn's own label, Tzadik Records, in 2001 and contains music that Zorn wrote and recorded for the documentary film "In the Mirror of
Maya Deren" on the life and work of Maya Deren directed by Martina Kudlácek.
</item 6>

<item 7>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 7>

<item 8>
Maya Deren Award
The American Film Institute Award for Independent Film and Video Artists, subtitled and generally known as the Maya Deren Award, was an award presented to
filmmakers and video artists by the American Film Institute to honor independent filmmaking.

 Named for the avant-garde experimental film artist Maya Deren, it was given from 1986 through 1996.
</item 8>

<item 9>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 9>

<item 10>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 10>
=== END ITEM id=5826af0e73fa17cc ===

=== ITEM id=67f8cc52e056a350 ===
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
Charkhi Dadri mid-air collision
The Charkhi Dadri mid-air collision occurred on 12 November 1996 over the village of Charkhi Dadri, to the west of New Delhi, India.

 The aircraft involved were a Saudi Arabian Airlines Boeing 747-100B en route from Delhi to Dhahran, Saudi Arabia, and a Kazakhstan Airlines Ilyushin Il-76 en
route from Chimkent, Kazakhstan, to Delhi.

 The crash killed all 349 people on board both planes, making it the world's deadliest mid-air collision, the deadliest aviation accident to occur in India, and
the third-deadliest aircraft accident in the history of aviation, behind only the Tenerife airport disaster and Japan Airlines Flight 123.
</item 1>

<item 2>
Iran Air Tours
Iran Air Tours is an airline based in Tehran, Iran.

 The airline mainly operates under the English name Iran Airtour (i.e. without the plural "s") as written in Persian (Persian: ایران ایر تور‎ ‎ ).

 This error is due to confusion between different pluralization rules in English and Persian.

 Iran Air Tours was established as a subsidiary of Iran Air and operates scheduled domestic services and international services in the Middle East as well as
Eastern Europe, as well as charter services.
</item 2>

<item 3>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 3>

<item 4>
1955 Cincinnati mid-air collision
The 1955 Cincinnati mid-air collision occurred on January 12, 1955, when a Trans World Airlines Martin 2-0-2 on takeoff from Boone County Airport (now the
Cincinnati/Northern Kentucky International Airport) collided in mid-air with a privately owned Douglas DC-3 that had entered the airport's control space without
proper clearance.

 There were no survivors.
</item 4>

<item 5>
1963 Aeroflot Tupolev Tu-124 Neva river ditching
The 1963 Aeroflot Tupolev Tu-124 Neva river ditching was a water landing by a Tupolev Tu-124 of the Soviet state airline "Aeroflot" (Moscow division).

 The aircraft took off from Tallinn-Ülemiste Airport (TLL) at 08:55 on 21 August with 45 passengers and 7 crew on board.

 The aircraft (registration number SSSR-45021) was built in 1962 and was scheduled to fly to Moscow–Vnukovo (VKO) under the command of 27-year-old captain
Victor Mostovoy.

 After takeoff the nose gear did not retract.

 Ground control diverted the flight to Leningrad (LED) – because of fog at Tallinn.
</item 5>

<item 6>
United Airlines Flight 736
United Airlines Flight 736 was a daily U.S. transcontinental passenger flight operated by United Airlines that crashed on April21, 1958, following a mid-air
collision.

 The aircraft assigned to Flight 736, a Douglas DC-7 airliner carrying 47 persons, was flying at cruise altitude above Clark County, Nevada, en route to a
stopover at Denver, Colorado, when it was struck by a United States Air Force fighter jet crewed by two pilots.

 The collision occurred at 8:30 a.m. in clear weather within a major commercial airway; both aircraft fell out of control from 21000 ft and crashed into
unpopulated desert terrain southwest of Las Vegas, Nevada.
</item 6>

<item 7>
Traffic collision avoidance system
A traffic collision avoidance system or traffic alert and collision avoidance system (both abbreviated as TCAS, and pronounced tee-kas) is an aircraft collision
avoidance system designed to reduce the incidence of mid-air collisions between aircraft.

 It monitors the airspace around an aircraft for other aircraft equipped with a corresponding active transponder, independent of air traffic control, and warns
pilots of the presence of other transponder-equipped aircraft which may present a threat of mid-air collision (MAC).

 It is a type of airborne collision avoidance system mandated by the International Civil Aviation Organization to be fitted to all aircraft with a maximum
take-off mass (MTOM) of over 5700 kg or authorized to carry more than 19 passengers.

 CFR 14, Ch I, part 135 requires that TCAS I is installed for aircraft with 10-30 passengers and TCAS II for aircraft with more than 30 passengers.
</item 7>

<item 8>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 8>

<item 9>
1993 Auckland mid-air collision
The 1993 Auckland mid-air collision was an aircraft accident in New Zealand.

 It occurred on 26 November 1993, when two aircraft operated by Airwork, under contract to the New Zealand Police, collided and crashed in central Auckland.

 The mid-air collision of the Aérospatiale TwinStar helicopter and Piper Archer aeroplane resulted in the deaths of all four occupants – a civilian Airwork
pilot on each aircraft and two New Zealand Police officers on the helicopter.

 The accident occurred in daylight with excellent visibility, in uncontrolled airspace (class G), with both aircraft flying under visual flight rules.

 Both the helicopter and aeroplane were operated by Airwork (NZ), and working under contract to the New Zealand Police at the time of the accident.
</item 9>

<item 10>
1993 Tehran mid-air collision
The 1993 Tehran mid-air collision occurred just outside the Iranian capital city of Tehran on Monday, February 8, 1993, killing all 133 people on board both
aircraft involved—an Iran Air Tours Tupolev Tu-154M, registered EP-ITD, and an Islamic Republic of Iran Air Force (IRIAF) Sukhoi Su-24 attack aircraft.
</item 10>
=== END ITEM id=67f8cc52e056a350 ===

=== ITEM id=42e5f85f79a418ae ===
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
Pulaski's Legion
Pulaski's Legion was raised on March 28, 1778 at Baltimore, Maryland under the command of Kazimierz Pułaski for service with the Continental Army during the
American Revolutionary War.

 The Legion consisted of one troop of lancers, two troops of dragoons and 200 light infantry soldiers.

 It was one of the few cavalry regiments in the American Continental Army.
</item 7>

<item 8>
Benjamin Logan
Benjamin Logan (c.1742 – December 11, 1802) was an American pioneer, soldier, and politician from Shelby County, Kentucky.

 As colonel of the Kentucky County, Virginia militia during the American Revolutionary War, he was second-in-command of all the militia in Kentucky.

 He was also a leader in Kentucky's efforts to become a state.

 His brother, John Logan, was the first state treasurer of Kentucky.
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
=== END ITEM id=42e5f85f79a418ae ===

=== ITEM id=1f2c7188a94cf829 ===
Gold sentences and facts:
Sentence 0: The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.
  Fact 0.0: The Straight Story is a 1999 film.
  Fact 0.1: The Straight Story is internationally co-produced.
  Fact 0.2: The Straight Story is a biographical road drama film.
  Fact 0.3: The Straight Story was directed by David Lynch.
Sentence 1: Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.
  Fact 1.0: Frozen is a 2013 film.
  Fact 1.1: Frozen is American.
  Fact 1.2: Frozen is a 3D computer-animated film.
  Fact 1.3: Frozen is a musical fantasy film.
  Fact 1.4: Frozen was produced by Walt Disney Animation Studios.
  Fact 1.5: Frozen was released by Walt Disney Pictures.

Retrieved items:
<item 1>
The Blizzard (1923 film)
The Blizzard is a 1923 Swedish drama film directed by Mauritz Stiller, starring Einar Hanson, Mary Johnson, Pauline Brunius and Hugo Björne.

 The film's original Swedish title is Gunnar Hedes saga, which means "The story of Gunnar Hede".

 The narrative revolves a student who tries to save his family's mansion which is facing bankruptcy.

 The film is loosely based on the Selma Lagerlöf novel "The Tale of a Manor".
</item 1>

<item 2>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 2>

<item 3>
The Future of Oil: A Straight Story of the Canadian Oil Sands
The Future of Oil: A Straight Story of the Canadian Oil Sands
</item 3>

<item 4>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 4>

<item 5>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 5>

<item 6>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 6>

<item 7>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 7>

<item 8>
Woman's film
The woman's film is a film genre which includes women-centered narratives, female protagonists and is designed to appeal to a female audience.

 Woman's films usually portray "women's concerns" such as problems revolving around domestic life, the family, motherhood, self-sacrifice, and romance.

 These films were produced from the silent era through the 1950s and early 1960s, but were most popular in the 1930s and 1940s, reaching their zenith during
World War II.

 Although Hollywood continued to make films characterized by some of the elements of the traditional woman's film in the second half of the 20th century, the
term itself disappeared in the 1960s.

 The work of directors George Cukor, Douglas Sirk, Max Ophüls, and Josef von Sternberg has been associated with the woman's film genre.

 Joan Crawford, Bette Davis, and Barbara Stanwyck were some of the genre's most prolific stars.
</item 8>

<item 9>
The Straight Story
The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.

 The film was edited and produced by Mary Sweeney, Lynch's longtime partner and co-worker.

 She co-wrote the script with John E. Roach.

 The film is based on the true story of Alvin Straight's 1994 journey across Iowa and Wisconsin on a lawn mower.

 Alvin (Richard Farnsworth) is an elderly World War II veteran who lives with his daughter Rose (Sissy Spacek), a kind woman with an intellectual disability.

 When he hears that his estranged brother Lyle (Harry Dean Stanton) has suffered a stroke, Alvin makes up his mind to go visit him and hopefully make amends
before he dies.

 Because Alvin's legs and eyes are too impaired for him to receive a driving license, he hitches a trailer to his recently purchased thirty-year-old John Deere
110 Lawn Tractor, having a maximum speed of about 5 miles per hour, and sets off on the 240 mile journey from Laurens, Iowa to Mount Zion, Wisconsin.
</item 9>

<item 10>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 10>
=== END ITEM id=1f2c7188a94cf829 ===

=== ITEM id=9983fbdc41b50f8c ===
Gold sentences and facts:
Sentence 0: Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.
  Fact 0.0: Wangaratta Wahine is an album by the Australian band The Captain Matchbox Whoopee Band.
  Fact 0.1: Wangaratta Wahine was released in 1974.
  Fact 0.2: Wangaratta Wahine was the second album of The Captain Matchbox Whoopee Band.
Sentence 1:  He was declared an Australian Living Treasure by the National Trust of Australia in 1999.
  Fact 1.0: The subject of this sentence was declared an Australian Living Treasure by the National Trust of Australia.
  Fact 1.1: The subject of this sentence received this declaration in 1999.

Retrieved items:
<item 1>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 1>

<item 2>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 2>

<item 3>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 3>

<item 4>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 4>

<item 5>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 5>

<item 6>
Graph query row:
album: Wangaratta Wahine
relationship: CREATED_BY
band: The Captain Matchbox Whoopee Band
</item 6>

<item 7>
Wangaratta Wahine
Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.

 It was recorded at top recording studio Armstrong Studios in Melbourne and the cover art was by famed Australian artist Michael Leunig.
</item 7>

<item 8>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 8>

<item 9>
He Will Have His Way
He Will Have His Way is a compilation album featuring male Australian and New Zealand musicians performing songs written by Neil Finn and Tim Finn (The Finn
Brothers), who are best known as members of Split Enz and Crowded House.

 It was released on 12 November 2010 and is a sequel to "She Will Have Her Way", a 2005 album featuring female Australian and New Zealand musicians.

 The two albums were later released together with extra tracks as the double CD "They Will Have Their Way".
</item 9>

<item 10>
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 10>
=== END ITEM id=9983fbdc41b50f8c ===

=== ITEM id=4c78ef0aaa40b07f ===
Gold sentences and facts:
Sentence 0:  It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
  Fact 0.0: The book is a companion book to 'A Wrinkle in Time'.
  Fact 0.1: The book is part of the Time Quintet.
Sentence 1: The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
  Fact 1.0: The Time Quintet is a fantasy/science fiction series.
  Fact 1.1: The Time Quintet is a series of five young adult novels.
  Fact 1.2: The Time Quintet was written by Madeleine L'Engle.

Retrieved items:
<item 1>
A Wrinkle in Time (2018 film)
A Wrinkle in Time is an upcoming American science-fiction fantasy adventure film directed by Ava DuVernay and written by Jennifer Lee.

 It is based on the 1963 novel of the same name by Madeleine L'Engle.

 The film stars Oprah Winfrey, Reese Witherspoon, Mindy Kaling, Storm Reid, Zach Galifianakis and Chris Pine.

 Principal photography on the film began on November 2, 2016, in Los Angeles, California.
</item 1>

<item 2>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 2>

<item 3>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 3>

<item 4>
The Four Winds of Love
The Four Winds of Love is the overall title for a series of six novels written by Compton Mackenzie, "The East Wind of Love" (1937), "The South Wind of Love"
(1938), "The West Wind of Love" (1940), "West to North" (1942), "The North Wind of Love, Book 1" (1944) and "The North Wind of Love, Book 2" (1945), which taken
together constitute a major fictional chronicle of the first forty years of the twentieth century.

 The main protagonist of the hexalogy is the semi-autobiographical character of John Ogilvie.
</item 4>

<item 5>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 5>

<item 6>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 6>

<item 7>
Time Quintet
The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
</item 7>

<item 8>
Kim Edwards
Kim Edwards (born May 4, 1958) is an American author and educator.

 She was born in Killeen, Texas, grew up in the Finger Lakes region of upstate New York, and graduated from Colgate University and The University of Iowa, where
she earned an MFA in fiction and an MA in linguistics.

 She is the author of a story collection, "The Secrets of a Fire King", which was a finalist for the PEN/Hemingway Award; her stories have been published in
"The Paris Review, Story, Ploughshares, Zoetrope", and many other periodicals.

 She has received many awards for the short story as well, including a Pushcart Prize, the National Magazine Award, the Nelson Algren Award, and inclusion in
both "The Best American Short Stories" and the Symphony Space program ‘Selected Shorts.’

 She is the recipient of a Whiting Writers’ Award, as well as grants from the Pennsylvania and Kentucky Arts Councils, the Kentucky Foundation for Women, and
the National Endowment for the Arts.
</item 8>

<item 9>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 9>

<item 10>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 10>
=== END ITEM id=4c78ef0aaa40b07f ===

=== ITEM id=6f54acdb48497ed9 ===
Gold sentences and facts:
Sentence 0: Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).
  Fact 0.0: Dictyosperma is a monotypic genus of flowering plant.
  Fact 0.1: Dictyosperma is in the palm family.
  Fact 0.2: Dictyosperma is found in the Mascarene Islands in the Indian Ocean.
  Fact 0.3: Dictyosperma is found in Mauritius.
  Fact 0.4: Dictyosperma is found in Réunion.
  Fact 0.5: Dictyosperma is found in Rodrigues.
Sentence 1: The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in 1810.
  Fact 1.0: The genus Huernia is in the family Apocynaceae.
  Fact 1.1: The genus Huernia is in the subfamily Asclepiadoideae.
  Fact 1.2: The genus Huernia consists of stem succulents.
  Fact 1.3: The stem succulents of the genus Huernia are from Eastern Africa.
  Fact 1.4: The stem succulents of the genus Huernia are from Southern Africa.
  Fact 1.5: The genus Huernia was first described as a genus in 1810.

Retrieved items:
<item 1>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 1>

<item 2>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 2>

<item 3>
Harrimanella
Harrimanella is a genus of flowering plant in the heath family Ericaceae, with a single species, Harrimanella hypnoides, also known as moss bell heather.

 It was originally named Cassiope hypnoides by Linnaeus (1737) in his Flora Lapponica, but "Harrimanella hypnoides" is now the accepted name at ITIS.

 The species name "hypnoides" means 'like "Hypnum" ', which is a genus mosses.
</item 3>

<item 4>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 4>

<item 5>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 5>

<item 6>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 6>

<item 7>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 7>

<item 8>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 8>

<item 9>
Hericium
Hericium is a genus of edible mushrooms in the Hericiaceae family.

 Species in this genus are white and fleshy and grow on dead or dying wood; fruiting bodies resemble a mass of fragile icicle-like spines that are suspended
from either a branched supporting framework or from a tough, unbranched cushion of tissue.

 This distinctive structure has earned "Hericium" species a variety of common names—monkey's head, lion's mane, and bear's head are examples.

 Taxonomically, this genus was previously placed within the order Aphyllophorales, but recent molecular studies now place it in the Russulales.
</item 9>

<item 10>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 10>
=== END ITEM id=6f54acdb48497ed9 ===

=== ITEM id=164d4a9e26951e79 ===
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
Grampa Simpson
Abraham Jedediah Simpson II, often known as Grampa, is a fictional character in the animated television series "The Simpsons".

 He made his first appearance in the episode entitled "Grampa and the Kids", a Simpsons short on "The Tracey Ullman Show".

 Voiced by Dan Castellaneta, he is the father of Homer Simpson and the grandfather of Bart, Lisa and Maggie Simpson.

 In the 1000th issue of "Entertainment Weekly", Grampa was selected as the Grandpa for "The Perfect TV Family".
</item 1>

<item 2>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 2>

<item 3>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 3>

<item 4>
Good Night (The Simpsons short)
"Good Night" (also known as "Good Night Simpsons") is the first of forty-eight Simpsons shorts that appeared on the variety show "The Tracey Ullman Show".

 It originally aired on the Fox network in the United States on April 19, 1987, during the third episode of "The Tracey Ullman Show" and marks the first
appearance of the Simpson family — Homer, Marge, Bart, Lisa, and Maggie — on television.

 After three seasons on Tracey Ullman, the shorts would be adapted into the animated show "The Simpsons".

 "Good Night" has since been aired on the show in the episode "The Simpsons 138th Episode Spectacular" (in its entirety), along with several other Ullman
shorts, and is one of the few shorts to ever be released on DVD, being included in the Season 1 DVD set.
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
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 7>

<item 8>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 8>

<item 9>
Girly Edition
"Girly Edition" is the twenty-first episode in the ninth season of the American animated television series "The Simpsons".

 It originally aired on the Fox network in the United States on April 19, 1998.

 In the episode, Lisa and Bart Simpson must co-anchor a new news program, though when Bart is seen as a more successful news anchor, Lisa becomes jealous and
seeks revenge.

 Meanwhile, in the subplot, Homer Simpson gets a monkey helper because of his laziness.

 "Girly Edition" was the first episode written by Larry Doyle and was directed by Mark Kirkland.

 Much of the subplot was inspired by the film "Monkey Shines".

 Critics gave the episode positive reviews and it was well received by Lisa's voice actress Yeardley Smith.
</item 9>

<item 10>
I, (Annoyed Grunt)-bot
"I, (Annoyed Grunt)-Bot", also known as ""I, D'oh-Bot"", is the ninth episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It originally aired on the Fox network in the United States on January 11, 2004.

 This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's
"Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
</item 10>
=== END ITEM id=164d4a9e26951e79 ===
