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

=== ITEM id=c73542b11cac4806 ===
Gold sentences and facts:
Sentence 0: The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy Schaal and Henry Gibson.
  Fact 0.0: The 'Burbs is a 1989 American film.
  Fact 0.1: The 'Burbs is a comedy thriller film.
  Fact 0.2: The 'Burbs was directed by Joe Dante.
  Fact 0.3: The 'Burbs stars Tom Hanks.
  Fact 0.4: The 'Burbs stars Bruce Dern.
  Fact 0.5: The 'Burbs stars Carrie Fisher.
  Fact 0.6: The 'Burbs stars Rick Ducommun.
  Fact 0.7: The 'Burbs stars Corey Feldman.
  Fact 0.8: The 'Burbs stars Wendy Schaal.
  Fact 0.9: The 'Burbs stars Henry Gibson.
Sentence 1: Richard "Rick" Ducommun (July 3, 1952 – June 12, 2015) was a Canadian actor, comedian and writer who appeared in films and television.
  Fact 1.0: Rick Ducommun was born on July 3, 1952.
  Fact 1.1: Rick Ducommun died on June 12, 2015.
  Fact 1.2: Rick Ducommun was a Canadian actor.
  Fact 1.3: Rick Ducommun was a Canadian comedian.
  Fact 1.4: Rick Ducommun was a Canadian writer.
  Fact 1.5: Rick Ducommun appeared in films.
  Fact 1.6: Rick Ducommun appeared in television.

Retrieved items:
<item 1>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 1>

<item 2>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 2>

<item 3>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 3>

<item 4>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 4>

<item 5>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 5>

<item 6>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 6>

<item 7>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 7>

<item 8>
Vegas Vacation
Vegas Vacation is a 1997 American comedy film directed by Stephen Kessler.

 It is the fourth installment in "National Lampoon"’s "Vacation" film series, and was written by Elisa Bell, based on a story by Bell and Bob Ducsay.

 The film stars Chevy Chase, Beverly D'Angelo and Randy Quaid, with Ethan Embry and Marisol Nichols as Griswold children Rusty and Audrey.

 The film opened at #4 at the box office and grossed over $36.4 million domestically.

 "Vegas Vacation" became the first theatrical "Vacation" film not to carry the National Lampoon label or a screenwriting credit from John Hughes.
</item 8>

<item 9>
Araby Lockhart
Araby Lockhart (born December 4, 1926) is a Canadian stage and television actress, best known for her performances in the films "Capote" and "Police Academy"
and her stage performances as a member of Hart House Theatre and the Straw Hat Players.

 Lockhart has also served as President of the Actors' Fund of Canada.
</item 9>

<item 10>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 10>
=== END ITEM id=c73542b11cac4806 ===

=== ITEM id=bb68618e960e6a84 ===
Gold sentences and facts:
Sentence 0: Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist, pianist, and primary songwriter for the Christian rock band Relient K.
  Fact 0.0: Matthew Arnold 'Matt' Thiessen was born August 12, 1980.
  Fact 0.1: Matt Thiessen is a Canadian-American musician, singer, and songwriter.
  Fact 0.2: Matt Thiessen is known for being co-founder of the Christian rock band Relient K.
  Fact 0.3: Matt Thiessen is known for being lead singer of the Christian rock band Relient K.
  Fact 0.4: Matt Thiessen is known for being guitarist of the Christian rock band Relient K.
  Fact 0.5: Matt Thiessen is known for being pianist of the Christian rock band Relient K.
  Fact 0.6: Matt Thiessen is known for being primary songwriter for the Christian rock band Relient K.
Sentence 1: George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.
  Fact 1.0: George Thorogood was born February 24, 1950.
  Fact 1.1: George Thorogood is an American musician, singer, and songwriter.
  Fact 1.2: George Thorogood is from Wilmington, Delaware.

Retrieved items:
<item 1>
More George Thorogood and The Destroyers
More George Thorogood and the Destroyers is the fourth album by George Thorogood and The Destroyers, released in 1980.

 An alternate name for the album is I'm Wanted.
</item 1>

<item 2>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 2>

<item 3>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 3>

<item 4>
Better Than the Rest
Better Than the Rest is the third album (mini-album) of songs by George Thorogood and the Destroyers, recorded in 1974 and released in 1979.

 At the time the songs were recorded, Thorogood was an unknown artist who was not signed to a record label.

 In 1979, after Thorogood had released his first two official albums on Rounder Records, "Better Than The Rest" was released by MCA.

 In 1986, the songs from this album were released on compact disc with the title "Nadine" using a different track order.
</item 4>

<item 5>
“Nadine” (1986) is a reordered CD reissue of George Thorogood and the Destroyers’ “Better Than the Rest” (1979), made from Thorogood’s first, 1974 recordings.
Thorogood, a Wilmington-born boogie-blues musician, is known for “Bad to the Bone” and blues covers. The text also notes the band’s 1977 debut, 1980 “More
George Thorogood” (“I’m Wanted
</item 5>

<item 6>
2120 South Michigan Ave.
2120 South Michigan Ave. is the fifteenth studio album by George Thorogood and the Destroyers.

 It was released on June 14, 2011 on the Capitol Records label.

 The album peaked at #2 on the "Billboard" Top Blues Albums chart.

 The title refers to the address of the offices and recording studios of Chess Records in Chicago.

 The album contains ten covers of songs recorded on Chess Records by artists such as Chuck Berry, Bo Diddley, Willie Dixon, and Muddy Waters; plus a cover of
The Rolling Stones' instrumental "2120 South Michigan Avenue" and two original songs about Chess Records artists.

 Capitol Records approached Thorogood with the idea for the album and selected most of the songs.

 Buddy Guy and Charlie Musselwhite perform on the album, although their work was added after primary recording was complete.
</item 6>

<item 7>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 7>

<item 8>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 8>

<item 9>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 9>

<item 10>
George Thorogood and the Destroyers (album)
George Thorogood and the Destroyers is the self-titled debut album by American blues rock band George Thorogood and the Destroyers, released in 1977.

 Consisting mostly of covers of blues hits, it included a medley of John Lee Hooker's "House Rent Boogie" and "One Bourbon, One Scotch, One Beer", the latter a
song written by Rudy Toombs for Amos Milburn, and later covered by Hooker.
</item 10>
=== END ITEM id=bb68618e960e6a84 ===

=== ITEM id=184c764dfea61d34 ===
Gold sentences and facts:
Sentence 0: Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.
  Fact 0.0: Andrew and Jeremy Get Married is a 2004 film.
  Fact 0.1: Andrew and Jeremy Get Married is a British documentary film.
  Fact 0.2: Andrew and Jeremy Get Married was written by Don Boyd.
  Fact 0.3: Andrew and Jeremy Get Married was directed by Don Boyd.
  Fact 0.4: Andrew and Jeremy Get Married was made for the BBC.
Sentence 1: The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.
  Fact 1.0: The Last Lions is a 2011 film.
  Fact 1.1: The Last Lions is an African nature documentary film.
  Fact 1.2: The Last Lions is by the National Geographic Society.
  Fact 1.3: The Last Lions was videotaped by Dereck and Beverly Joubert.
  Fact 1.4: The Last Lions was directed by Dereck and Beverly Joubert.

Retrieved items:
<item 1>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 1>

<item 2>
The Wolf of Wall Street (2013 film)
The film marks the director's fifth collaboration with DiCaprio, after "Gangs of New York" (2002), "The Aviator" (2004), "The Departed" (2006) and "Shutter
Island" (2010), as well as his second collaboration with Winter after the television series "Boardwalk Empire" (2010–14).
</item 2>

<item 3>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 3>

<item 4>
Get Married 3
Get Married 3 is an Indonesian romantic comedy directed by Monty Tiwa and released in 2011.

 A sequel to Hanung Bramantyo's "Get Married" and "Get Married 2", it stars Nirina Zubir and Fedi Nuril as a married couple attempting to raise their triplets
while under intense pressure from their family and friends.

 The film was a commercial success and received favourable reviews in "The Jakarta Post" and "Suara Karya".
</item 4>

<item 5>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 5>

<item 6>
Richard T. Jones
Richard Timothy Jones (born January 16, 1972), known professionally as Richard T. Jones, is an American actor.

 Jones is best known for his portrayals of Laveinio in the dramatic film "The Wood" and Mike of the dramatic films "Why Did I Get Married?

" and "Why Did I Get Married Too?

".
</item 6>

<item 7>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 7>

<item 8>
Two Brides and a Baby
Two Brides and a Baby is a 2011 Nigerian romantic drama film directed by Teco Benson, starring Keira Hewatch, Kalu Ikeagwu, OC Ukeje, Chelsea Eze, Stella
Damasus-Aboderin and Okey Uzoeshi.

 It premiered on November 17, 2011.

 It received awards and nominations at Africa Movie Academy Awards, Best of Nollywood Awards and Africa Magic Viewers Choice Awards.
</item 8>

<item 9>
Andrew and Jeremy Get Married
Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.

 It tells the story of two Englishmen, Andrew Thomas and Jeremy Trafford, as they plan for their commitment ceremony.

 Originally commissioned for the "BBC Storyville" series, the film premiered at the 2004 Toronto International Film Festival.
</item 9>

<item 10>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 10>
=== END ITEM id=184c764dfea61d34 ===

=== ITEM id=137f49239de53c39 ===
Gold sentences and facts:
Sentence 0:  He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.
  Fact 0.0: The person finished second in voting for the Heisman Trophy in 1956.
  Fact 0.1: Paul Hornung finished first in that Heisman Trophy voting in 1956.
Sentence 1: Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.
  Fact 1.0: Paul Vernon Hornung was born on December 23, 1935.
  Fact 1.1: Paul Vernon Hornung is nicknamed "The Golden Boy".
  Fact 1.2: Paul Vernon Hornung is a former professional American football player.
  Fact 1.3: Paul Vernon Hornung is a Hall of Fame running back.
  Fact 1.4: Paul Vernon Hornung played for the Green Bay Packers of the National Football League (NFL).
  Fact 1.5: Paul Vernon Hornung played for the Green Bay Packers from 1957 to 1966.
  Fact 1.6: Paul Vernon Hornung won four NFL titles.
  Fact 1.7: Paul Vernon Hornung won the first Super Bowl.

Retrieved items:
<item 1>
Johnny Majors
John Terrill Majors (born May 21, 1935) is a former American football player and coach.

 A standout halfback at the University of Tennessee, he was an All-American in 1956 and a two-time winner of the Southeastern Conference Most Valuable Player
award, in 1955 and 1956.

 He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.

 Majors served as the head football coach at Iowa State University (1968–1972), the University of Pittsburgh (1973–1976, 1993–1996), and Tennessee (1977–1992),
compiling a career college football record of 185–137–10.

 His 1976 Pittsburgh squad won a national championship after capping a 12–0 season with a victory in the Sugar Bowl.

 Majors was inducted into the College Football Hall of Fame as a player in 1987.
</item 1>

<item 2>
Heisman Trophy
The Heisman Memorial Trophy (usually known colloquially as the Heisman Trophy or The Heisman), is awarded annually to the most outstanding player in college
football in the United States whose performance best exhibits the pursuit of excellence with integrity.

 Winners epitomize great ability combined with diligence, perseverance, and hard work.

 It is presented by the Heisman Trophy Trust in early December before the postseason bowl games.
</item 2>

<item 3>
Frank Eliscu
Frank Eliscu (1912–1996) was an American sculptor and art teacher who designed and created the Heisman Memorial Football Trophy in 1935 when he was only 20
years old.

 The first Heisman Trophy, a strong young bull of a football player cast in bronze, was presented to a college football player in 1935, and is considered one of
the greatest honors a college athlete can receive.

 Over the years the Heisman Committee has paid tribute to Eliscu and his creation several times, and in 1985, the Heisman Committee invited him to speak at the
50th Anniversary Heisman ceremonies.
</item 3>

<item 4>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 4>

<item 5>
1976 NCAA Division I football season
The 1976 NCAA Division I football season ended with a championship for the Panthers of the University of Pittsburgh.

 Led by head coach Johnny Majors (voted the AFCA Coach of the Year), the Pitt Panthers brought a college football championship to the home of the defending pro
football champions, the Steelers.

 Pitt also had the Heisman Trophy winner, Tony Dorsett; the Panthers had been ranked ninth in the preseason AP poll.
</item 5>

<item 6>
Heisman curse
Although many Heisman winners have not enjoyed success at the professional level, including players like Matt Leinart, Andre Ware, Jason White, Rashaan Salaam,
Eric Crouch, Ty Detmer, Troy Smith and Gino Torretta, proponents of the "curse" rarely cite highly successful players such as Barry Sanders, Charles Woodson,
Eddie George, Tim Brown, Bo Jackson, Marcus Allen, Earl Campbell, and Tony Dorsett among the notables.
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
Charlie Ward
Charlie Ward Jr. (born October 12, 1970) is a retired American professional NBA basketball player, college football Heisman Trophy winner, Davey O'Brien Award
winner and a Major League Baseball draftee.

 Despite his NCAA football success, Ward was one of the very few players who won a Heisman trophy but was not drafted in the NFL draft.

 He won the College Football National Championship with the Florida State University Seminoles.

 Ward played several years with the New York Knicks and started in the NBA Finals.

 He was inducted in the College Football Hall of Fame in 2006.

 An avid tennis player, Ward also displayed his skills at the Arthur Ashe Tennis Tournament in 1994.
</item 8>

<item 9>
1997 Tennessee Volunteers football team
However, the Vols' defense could not stop Nebraska's rushing attack, giving up more than 400 yards on the ground in a 42–17 loss.

 As a senior, Manning won numerous awards.

 He was a consensus first-team All-American and won the Maxwell Award, the Davey O'Brien Award, the Johnny Unitas Award, and the Best College Football Player
ESPY Award, among others.

 However, he did not win the Heisman Trophy, finishing runner-up to Charles Woodson, a CB from Michigan, and the only defensive player ever to win the Heisman
Trophy.
</item 9>

<item 10>
Graph query row:
person: Paul Vernon Hornung
birth_date: December 23, 1935
heisman_trophy: None
trophy_connection: None
connected_person: None
person_connection: None
</item 10>
=== END ITEM id=137f49239de53c39 ===

=== ITEM id=7a287654cfb52872 ===
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
Graph query row:
entity: Inner Secrets
relationship: CREATED_BY
related: Santana
</item 1>

<item 2>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 2>

<item 3>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 3>

<item 4>
Inner Secrets
Inner Secrets is the ninth studio album by Santana.

 It was released in 1978 and marks the start of the phase of Santana's career where he moved away from the fusion of Latin, jazz, rock and blues that marked his
previous records and began to move towards an album-oriented rock direction.

 As such, the album's quality is widely disputed among fans. "

Stormy" and "One Chain (Don't Make No Prison)" were both hit singles.

 In The Netherlands "Well All Right" was released as a single and reached #22 in the top 40.
</item 4>

<item 5>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 5>

<item 6>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 6>

<item 7>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 7>

<item 8>
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
</item 8>

<item 9>
Chicago 13
Chicago 13 is the eleventh studio album by the American band Chicago, released in 1979.

 The follow-up to "Hot Streets", "Chicago 13" is often critically disfavored.

 This would be the band's final release that features lead guitarist Donnie Dacus, who had followed late, founding lead guitarist, Terry Kath.

 All band members would contribute to the songwriting (one of only two albums where this is the case, with the other being "Chicago VII").
</item 9>

<item 10>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 10>
=== END ITEM id=7a287654cfb52872 ===

=== ITEM id=88ce3d86176471fe ===
Gold sentences and facts:
Sentence 0: Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
  Fact 0.0: Javier Alberto Frana was born on 25 December 1966.
  Fact 0.1: Javier Alberto Frana is a former tennis player.
  Fact 0.2: Javier Alberto Frana is from Argentina.
  Fact 0.3: Javier Alberto Frana is a current tennis commentator.
  Fact 0.4: Javier Alberto Frana is a tennis commentator for ESPN Latin America.
Sentence 1: Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and the U.S. National Championships.
  Fact 1.0: Thomaz Koch was born on May 11, 1945.
  Fact 1.1: Thomaz Koch was born in Porto Alegre.
  Fact 1.2: Thomaz Koch is a left-handed former tennis player.
  Fact 1.3: Thomaz Koch is from Brazil.
  Fact 1.4: Thomaz Koch was a quarter-finalist at the French Open.
  Fact 1.5: Thomaz Koch was a quarter-finalist at Wimbledon.
  Fact 1.6: Thomaz Koch was a quarter-finalist at the U.S. National Championships.

Retrieved items:
<item 1>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 1>

<item 2>
Javier Frana
Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
</item 2>

<item 3>
Thomaz Koch
Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and
the U.S. National Championships.
</item 3>

<item 4>
1994 Movistar Open – Singles
Javier Frana was the defending champion but lost in the quarterfinals to Alberto Berasategui.
</item 4>

<item 5>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 5>

<item 6>
1995 Nottingham Open – Singles
Javier Frana defeated Todd Woodbridge 7–6, 6–3 in the final to secure the title.
</item 6>

<item 7>
Franz Josef Och
Franz Josef Och (November 2, 1971) is a German computer scientist.

 He is currently working at Grail (an Illumina company) as Head of Data Science.

 Prior to this he was Chief Data Scientist at Human Longevity Inc., and earlier worked for Google as a Distinguished Research Scientist and head of machine
translation based at Google's Mountain View, California, headquarters south of San Francisco.
</item 7>

<item 8>
1993 Movistar Open – Singles
Javier Frana defeated Emilio Sánchez 7–5, 3–6, 6–3 to win the 1993 Movistar Open singles competition.
</item 8>

<item 9>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 9>

<item 10>
Adam František Kollár
Adam František Kollár de Keresztén (German: "Adam Franz Kollar von Keresztén" , Hungarian: "kereszténi Kollár Ádám Ferenc" ; 1718–1783) was a Slovak jurist,
Imperial-Royal Court Councilor and Chief Imperial-Royal Librarian, a member of Natio Hungarica in the Kingdom of Hungary, a historian, ethnologist, an
influential advocate of Empress Maria Theresa's Enlightened and centralist policies.

 His advancement of Maria Theresa's status in the Kingdom of Hungary as its apostolic ruler in 1772 was used as an argument in support of the subsequent
Habsburg annexations of Galicia and Dalmatia.

 Kollár is also credited with coining the term "ethnology" and providing its first definition in 1783.

 Some authors see him as one of the earliest pro-Slovak, pro-Slavic, and pan-Slavic activists in the Habsburg Monarchy.
</item 10>
=== END ITEM id=88ce3d86176471fe ===

=== ITEM id=d94894ec5dbff003 ===
Gold sentences and facts:
Sentence 0: Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.
  Fact 0.0: Ronkonkoma is a major railroad station.
  Fact 0.1: Ronkonkoma is a major transportation hub.
  Fact 0.2: Ronkonkoma is along the Main Line of the Long Island Rail Road.
  Fact 0.3: The Ronkonkoma railroad station is in Ronkonkoma, New York.
Sentence 1: Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.
  Fact 1.0: Ronkonkoma is a hamlet.
  Fact 1.1: Ronkonkoma is a census-designated place (CDP).
  Fact 1.2: Ronkonkoma is on Long Island.
  Fact 1.3: Ronkonkoma is in the Town of Islip.
  Fact 1.4: The Town of Islip is in New York, United States.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 2>

<item 3>
Long Island Electric Railway
The Long Island Electric Railway was a streetcar company operating in Brooklyn, Queens, and Nassau County, New York, United States between 1894 and 1926.

 The company was partially owned by the Long Island Consolidated Electric Companies, a holding company for the Long Island Rail Road and partially by August
Belmont and the Interborough Rapid Transit Company.

 It connected the east end of the Fulton Street El at Crescent Street station in City Line, Brooklyn with Jamaica, Queens, and ran from there to the Nassau
County line at Queens Village and to Far Rockaway, Queens via Nassau County.

 It also had a connection to Belmont Park.

 The New York and Long Island Traction Company used trackage rights over its line from Crescent Street to Queens Village.
</item 3>

<item 4>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 4>

<item 5>
North Creek Railroad Station
The North Creek Railroad Station is a historic railroad station complex located at North Creek, Warren County, New York.

 The complex consists of the railroad station, the freight house, round house, turntable, and horse barn.

 The station was built in 1874 and is a simple, rectangular, gable roofed building with a broad, overhanging strut-supported roof in the Stick-Eastlake style.

 Its exterior is covered with vertical boards.
</item 5>

<item 6>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 6>

<item 7>
Lacona Railroad Station and Depot
Lacona Railroad Station and Depot, also known as New York Central and Hudson River Railroad Station and Depot, is a historic railway depot located at Lacona in
Oswego County, New York.

 It was built in 1891 by the New York Central and Hudson River Railroad.

 It is a small rectangular, one story, gable ended structure.
</item 7>

<item 8>
Ronkonkoma (LIRR station)
Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.

 The station is the eastern terminus of the Ronkonkoma Branch and the western terminus of the out Branch.
</item 8>

<item 9>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 9>

<item 10>
Jamestown Gateway Train Station
The Jamestown Gateway Train Station, also known as the Jamestown Erie Railroad station, and the Erie-Lackawanna Railroad Station, is a historic train station
located at Jamestown in Chautauqua County, New York.

 Although no longer an active railroad station due to a lack of passenger service in the area after a restoration done in 2011 the building currently serves as
a bus transportation center and community space for Jamestown.

 The first train arrived at Jamestown on August 25, 1860 as part of the Atlantic and Great Western Railroad.
</item 10>
=== END ITEM id=d94894ec5dbff003 ===

=== ITEM id=6639f413fe7d6574 ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Nicolas Cage starred in a variety of films.
  Fact 0.1: Nicolas Cage starred in Valley Girl (1983).
  Fact 0.2: Nicolas Cage starred in Racing with the Moon (1984).
  Fact 0.3: Nicolas Cage starred in Birdy (1984).
  Fact 0.4: Nicolas Cage starred in Peggy Sue Got Married (1986).
  Fact 0.5: Nicolas Cage starred in Raising Arizona (1987).
  Fact 0.6: Nicolas Cage starred in Moonstruck (1987).
  Fact 0.7: Nicolas Cage starred in Vampire's Kiss (1989).
  Fact 0.8: Nicolas Cage starred in Wild at Heart (1990).
  Fact 0.9: Nicolas Cage starred in Fire Birds (1990).
  Fact 0.10: Nicolas Cage starred in Honeymoon in Vegas (1992).
  Fact 0.11: Nicolas Cage starred in Red Rock West (1993).
Sentence 1:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 1.0: The film is about a widowed 37-year-old Italian-American woman.
  Fact 1.1: Cher plays the widowed 37-year-old Italian-American woman.
  Fact 1.2: The widowed woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 1.3: The widowed woman's fiancé is played by Danny Aiello.
  Fact 1.4: The fiancé's younger brother is estranged.
  Fact 1.5: The fiancé's younger brother is hot-tempered.
  Fact 1.6: The fiancé's younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 1>

<item 2>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 2>

<item 3>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 3>

<item 4>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 4>

<item 5>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 5>

<item 6>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 6>

<item 7>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 7>

<item 8>
The entries cover films across action, comedy, romance, horror, and drama. Bruce Willis stars in titles including *Last Man Standing* (a credited *Yojimbo*
remake), *Unbreakable*, *Signs*, and *Die Hard with a Vengeance*. Shyamalan’s *Signs* explores faith and alien encounters; *Unbreakable* launches a superhero
trilogy. Other highlights include *The Whole Nine Yards*, *Striking Distance*, *Mercury Rising
</item 8>

<item 9>
The collection spans films and TV from 1947–2010: Bruce Willis’s dramas, action comedies and thrillers include *In Country* (Golden Globe-nominated), *Last Man
Standing* (a *Yojimbo* remake), *The Whole Nine Yards* and *Cop Out*. Other highlights: *The Bears and I*; teen comedies *Can’t Buy Me Love* and *The Night
Before*; *The Country Girl* (Oscar-winning screenplay
</item 9>

<item 10>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 10>
=== END ITEM id=6639f413fe7d6574 ===

=== ITEM id=2a8306a23edc4208 ===
Gold sentences and facts:
Sentence 0: Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.
  Fact 0.0: Summer Magic is a 1963 Walt Disney Productions film.
  Fact 0.1: Summer Magic stars Hayley Mills.
  Fact 0.2: Summer Magic stars Burl Ives.
  Fact 0.3: Summer Magic stars Dorothy McGuire.
  Fact 0.4: Summer Magic is a story about a Boston widow and her children.
  Fact 0.5: In Summer Magic, the Boston widow and her children take up residence in a small town in Maine.
Sentence 1: Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.
  Fact 1.0: Around the World in 80 Days is a 2004 American film.
  Fact 1.1: Around the World in 80 Days is an action-adventure comedy family film.
  Fact 1.2: Around the World in 80 Days is based on Jules Verne's novel of the same name.

Retrieved items:
<item 1>
Around the World (1956 song)
"Around the World" was the theme tune from the 1956 movie "Around the World in 80 Days" In the film, only an instrumental version of the song appeared, although
the vocal version has become by far the better known one.

 The song was written by Harold Adamson and Victor Young; Young died in 1956, several weeks after the film's release and he received the Academy Award for Best
Music, Scoring of a Dramatic or Comedy Picture posthumously.

 Young's orchestral version was a #13 hit on the Billboard charts in 1957.

 The recording by Bing Crosby was the B-side of the Victor Young version in 1957, on Festival SP45-1274 in Australia, and was a joint charting success.
</item 1>

<item 2>
Around the World in 80 Days (2004 film)
Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.

 It stars Jackie Chan, Steve Coogan and Cécile de France.

 The film is set in 19th-century Britain and centers on Phileas Fogg (Steve Coogan), here reimagined as an eccentric inventor, and his efforts to circumnavigate
the globe in 80 days.

 During the trip, he is accompanied by his Chinese valet, Passepartout (Jackie Chan).

 For comedic reasons, the film intentionally deviated wildly from the novel and included a number of anachronistic elements.

 With production costs of about $110 million and estimated marketing costs of $30 million, it earned $24 million at the U.S. box office and $72 million
worldwide, making it a box office flop.

 It was Arnold Schwarzenegger's last film before he took a hiatus from acting to become Governor of California until 2010's "The Expendables".
</item 2>

<item 3>
People Who Travel (1938 German-language film)
Travelling People (German:Fahrendes Volk) is a 1938 German drama film directed by Jacques Feyder and starring Hans Albers, Françoise Rosay and Camilla Horn.

 It is a circus film.

 It premiered in Hamburg on 1 July 1938.

 A Separate French-language version "People Who Travel" ("Les gens du voyage") was also released.

 While it was also directed by Feyder and starred Rosay, the rest of the cast were different.
</item 3>

<item 4>
Around the World in 80 Days (1956 film)
Around the World in 80 Days (sometimes spelled as Around the World in Eighty Days) is a 1956 American epic adventure-comedy film starring Cantinflas and David
Niven , produced by the Michael Todd Company and released by United Artists.
</item 4>

<item 5>
Around the World in Eighty Days
Around the World in Eighty Days (French: "Le tour du monde en quatre-vingts jours" ) is a classic adventure novel by the French writer Jules Verne, published in
1873.

 In the story, Phileas Fogg of London and his newly employed French valet Passepartout attempt to circumnavigate the world in 80 days on a £20,000 wager (the
approximate equivalent of £2 million in 2016) set by his friends at the Reform Club.

 It is one of Verne's most acclaimed works .
</item 5>

<item 6>
Summer Holiday (1963 film)
Summer Holiday is a British CinemaScope and Technicolor musical film featuring singer Cliff Richard.

 The film was directed by Peter Yates (his debut), produced by Kenneth Harper.

 The original screenplay was written by Peter Myers and Ronald Cass (who also wrote most of the song numbers and lyrics).

 The cast includes Lauri Peters, Melvyn Hayes, Teddy Green, Jeremy Bulloch, Una Stubbs, Pamela Hart, Jacqueline Daryl, Lionel Murton, Madge Ryan, David Kossoff,
Nicholas Phipps, Ron Moody and The Shadows.

 Herbert Ross choreographed the musical numbers.
</item 6>

<item 7>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 7>

<item 8>
Summer Magic
Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children
taking up residence in a small town in Maine.

 The film was based on the novel "Mother Carey's Chickens" by Kate Douglas Wiggin and was directed by James Neilson.

 This was the fourth of six film Mills did for Disney, and the young actress received a Golden Globe nomination for her work here.
</item 8>

<item 9>
A Boy Scout Around the World
A Boy Scout Around the World (Danish: "Jorden Rundt i 44 dage", literally: "Around the World in 44 Days") is a travel description published in October 1928 and
written by Danish Boy Scout and later actor Palle Huld at the age of 15 following his travel around the world in spring 1928.

 His trip was sponsored by a Danish newspaper and made on the occasion of the 100 birthday of Jules Verne a French author of adventure and science fiction.

 Palle Huld was chosen after having answered to an ad in the newspaper; applicants had to be boys, 15 years old, able to manage in English and German and of
good health.

 Like the characters in Jules Verne’s novel "Around the World in 80 days" he was only allowed to travel by land and sea, not by air.

 The travel (on first class) went from Denmark to Great Britain, across the Atlantic Ocean to Canada.

 From the American west coast he continued to Japan, China, Soviet Union, Poland, Germany and back to Denmark.
</item 9>

<item 10>
Around the World in 80 Days (1988 film)
Around the World in 80 Days is an Australian 48-minute direct-to-video animated film from Burbank Films Australia.

 It was originally released in 1988.

 The film is based on Jules Verne's classic French novel, "Around the World in 80 Days", first published in 1873, and was adapted by Leonard Lee.

 It was produced by Roz Phillips and featured original music by Simon Walker.

 The film imitated BRB Internacional's Spanish 1981 series, "La vuelta al mundo de Willy Fog", in its use of anthropomorphic animals in the human roles.

 The copyright in this film is now owned by Pulse Distribution and Entertainment and administered by digital rights management firm NuTech Digital.
</item 10>
=== END ITEM id=2a8306a23edc4208 ===

=== ITEM id=76df8094b14a692e ===
Gold sentences and facts:
Sentence 0: Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that confirmed Trevi's status as a gay icon.
  Fact 0.0: Todos Me Miran ('Everyone looks at me') is a single from the Mexican artist Gloria Trevi.
  Fact 0.1: Todos Me Miran reached number 32 on Latin charts.
  Fact 0.2: Todos Me Miran became a club anthem.
  Fact 0.3: Todos Me Miran confirmed Trevi's status as a gay icon.
Sentence 1: Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
  Fact 1.0: Gloria Trevi was born Gloria de los Angeles Trevino Ruiz.
  Fact 1.1: Gloria Trevi was born on February 15, 1968.
  Fact 1.2: Gloria Trevi is a Mexican singer.
  Fact 1.3: Gloria Trevi is a Mexican songwriter.

Retrieved items:
<item 1>
Tu Ángel de la Guarda
Tu Ángel de la Guarda ("Your guardian angel") is Gloria Trevi's second album, and it contained one of her signature songs and her most widely known hit, "Pelo
Suelto".

 It also contained other hit songs such as "Tu angel de la guarda", "Ya no", "Virgen de las virgenes", and "Hoy me ire de casa".

 This album was very controversial different from other artists' albums such as Lucero.

 "Virgen de las virgenes" mocked girls who said they were virgins but had actually lost their virginity, and "¡Ya no!"

 went against the machismo movement of México.

 Following the release of the album, Gloria was working hard in promoting radio, television and print media.

 Trevi first traveled abroad visiting American Union countries, Puerto Rico, Argentina, Colombia, Guatemala, Venezuela, Costa Rica, and received a Gold and
Platinum for high sales achieved.
</item 1>

<item 2>
Todos Los Romances
Todos Los Romances (English: "All the Romances" ) is a three-disc compilation album by Mexican singer Luis Miguel.

 Released on 11 August 1998 by WEA Latina, the record features the three previously released "Romance"-themed albums in which Miguel covered classic boleros in
each of them: "Romance" (1991), "Segundo Romance" (1994), and "Romances" (1997).

 An editor for AllMusic rated the album four of five stars.

 Commercially, "Todos Los Romances" peaked at number four in Spain and was certified double Platinum in the country.

 It also achieved Gold status in Argentina and peaked at number 12 on the "Billboard"'s Top Latin Albums in the United States.
</item 2>

<item 3>
Graph query row:
path: ['Gloria Trevi', 'Todos Me Miran']
relationships: ['CREATED_BY']
</item 3>

<item 4>
Mexico Espanol Airplay
Mexico Espanol Airplay is a record chart published weekly by "Billboard" magazine for Spanish language singles released in Mexico.

 According to "Billboard"' s electronic database, the first chart was published on October 1, 2011, with the track "Amor Clandestino" by Mexican band Maná at
number one; in the same year, fellow Mexican duo Jesse & Joy reached the top of the chart with their single "¡Corre!

", the first of their six number-one singles in the chart, the most for any act. "

¡Corre!"

 also won the Latin Grammy Awards for Record of the Year and Song of the Year in 2012.

 Starting on December 1, 2012, Mexican singer Thalía spent 10 non-consecutive weeks at number-one with "Manías", the first single from the album "Habítame
Siempre", which won the Lo Nuestro Award for Pop Album of the Year. "

Hoy Tengo Ganas de Ti" by Mexican artist Alejandro Fernández and American singer Christina Aguilera reached number-one in 2013, and also peaked at number four
in Spain.
</item 4>

<item 5>
Gloria Trevi
Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
</item 5>

<item 6>
No eres tú, soy yo
No eres tú, soy yo (English: It's not you, it's me ) is a 2010 Mexican romantic comedy directed by Alejandro Springall starring Eugenio Derbez, Alejandra Barros
and Martina García and based on the Argentinean film "No sos vos, soy yo" (2004), written by Juan Taratuto.

 It was produced by Matthias Eherenberg and filmed in Mexico City.
</item 6>

<item 7>
Todos Me Miran
Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that
confirmed Trevi's status as a gay icon.

 The song, as interpreted in the music video, is about a young man who dares to crossdress in spite of society's opinions.
</item 7>

<item 8>
Premio Lo Nuestro 2003
In the Regional Mexican, Pilar Montenegro took three awards for Regional Mexican Song of the Year, Pop Song of the Year ("Quitame Ese Hombre"), and for Best
Regional Mexican Female Artist.

 In the tropical genre, Celia Cruz took home with four great awards of the night for Best Salsa Performance, Best Tropical Female Artist, Best Tropical Song of
the year and Tropical Album of the Year.

 At the night, the greatest performance was a medley of top Latin hits from the last 15 years, performed by the artists that made them famous, including Vikki
Carr, Son by Four, Los Ilegales, La Mafia, Luis Enrique, Wilfrido Vargas and Olga Tañón.

 There was a great tribute to Celia Cruz by the world-famous salsa group "Fania All-Stars", of which Cruz was a member during the 1970s, reunited for an
exclusive performance that rocked the house.
</item 8>

<item 9>
De Contrabando
"De Contrabando" ("Smuggled") is a song written by Joan Sebastian.

 It was first performed by Mexican singer Imelda, and included on her album "Aparentemente", released in 1991.

 Six years later, was performed by fellow Mexican singer Zayda Y Los Culpables and released by Musart Records as the second single from her self-titled album in
1997.

 Mexican-American singer Jenni Rivera recorded a cover version for her studio album "Parrandera, Rebelde y Atrevida".

 "De Contrabando" was released as the second single from the album in 2006.

 In the United States, the song peaked at number fourteen on the "Billboard" Hot Latin Songs chart and number one on the "Billboard" Regional Mexican Songs
chart.

 The song was nominated for a Lo Nuestro Award for Regional Mexican Song of the Year at the 2007 Lo Nuestro Awards.

 Rivera was also recognized for Regional Mexican Female Artist of the Year.

 Rivera was also awarded Regional Mexican Airplay Song of the Year for at the 2007 Latin "Billboard" Music Awards.
</item 9>

<item 10>
La Trayectoria (Gloria Trevi album)
La Trayectoria (In English: The Trajectory) is a 2006 double live CD and multimedia DVD release by the Mexican singer Gloria Trevi.

 The DVD includes scenes from parts of the tour from the Palacio de los Deportes of the Mexico City and The Arena Monterrey and the CD includes four new songs
like: "Todos Me Miran" (They All Look At Me) is the first single compilation study of Mexican singer "Gloria Trevi", this is an unreleased song from the album.
</item 10>
=== END ITEM id=76df8094b14a692e ===

=== ITEM id=90f067bf765084af ===
Gold sentences and facts:
Sentence 0:  He is best known for his roles in "How To Be Indie" (2009–2011), "Wingin' It" (2010–2013), "" (2012–2013), and "Open Heart" (2015).
  Fact 0.0: The person described is best known for a role in 'How To Be Indie' (2009–2011).
  Fact 0.1: The person described is best known for a role in 'Wingin' It' (2010–2013).
  Fact 0.2: The person described is best known for a role in an untitled work (2012–2013).
  Fact 0.3: The person described is best known for a role in 'Open Heart' (2015).
Sentence 1: Open Heart is a 2015 American-Canadian mystery-drama television series produced by the Epitome Pictures unit of DHX Media in association with marblemedia.
  Fact 1.0: Open Heart is a 2015 television series.
  Fact 1.1: Open Heart is an American-Canadian television series.
  Fact 1.2: Open Heart is a mystery-drama television series.
  Fact 1.3: Open Heart was produced by the Epitome Pictures unit of DHX Media.
  Fact 1.4: Open Heart was produced in association with marblemedia.

Retrieved items:
<item 1>
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 1>

<item 2>
Wingin' It
Wingin' It (originally titled Angel on Campus prior to its debut) is a Canadian teen sitcom which aired on Family.

 The series was produced by Temple Street Productions in association with Family.

 It stars Demetrius Joyette and Dylan Everett.

 This show included guest-stars from other Family Channel shows such as "The Latest Buzz".

 It was announced that the series was renewed for a third season on June 13, 2011.

 Family announced that the series is not planned for a fourth season.
</item 2>

<item 3>
Open Heart (TV series)
Open Heart is a 2015 American-Canadian mystery-drama television series produced by the Epitome Pictures unit of DHX Media in association with marblemedia.

 It first aired on January 20, 2015, on TeenNick in the United States and previously on YTV; now currently on ABC Spark in Canada.
</item 3>

<item 4>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 4>

<item 5>
Dan Flannery
Arthur Daniel Flannery (born March 10, 1944) is an American actor from Evanston, Illinois.

 He has played roles in several movies, short films, and T.V. shows from a variety of genres including drama, action, and thriller.

 His filmography includes The Straight Story (1999), and Contagion (2011) and T.V. series such as Empire (2015 TV series) and Boss (2011-2012).
</item 5>

<item 6>
Everest (2015 film)
Everest is a 2015 British-American biographical adventure film directed and co-produced by Baltasar Kormákur, co-produced by Tim Bevan, Eric Fellner, Nicky
Kentish Barnes, Tyler Thompson and Brian Oliver and written by William Nicholson and Simon Beaufoy, adapted from Beck Weathers' memoir "Left for Dead: My
Journey Home from Everest" (2000).

 It stars an ensemble cast of Jason Clarke, Josh Brolin, John Hawkes, Robin Wright, Michael Kelly, Sam Worthington, Keira Knightley, Emily Watson, and Jake
Gyllenhaal.

 It is based on the real events of the 1996 Mount Everest disaster, and focuses on the survival attempts of two expedition groups, one led by Rob Hall (Clarke)
and the other by Scott Fischer (Gyllenhaal).
</item 6>

<item 7>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 7>

<item 8>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 8>

<item 9>
Dean Winchester
Dean Winchester is a fictional character and one of the two protagonists of the American drama television series "Supernatural".

 He is portrayed by Jensen Ackles.

 Other versions of the character having been portrayed by Hunter Brochu (toddler), Ridge Canipe (child), Nicolai Lawton-Giustra (pre-teen), Brock Kelly and
Dylan Everett (teen), and Chad Everett (elderly).
</item 9>

<item 10>
Dylan Everett
Dylan Everett (born January 24, 1995) is a Canadian actor.

 He is best known for his roles in "How To Be Indie" (2009–2011), "Wingin' It" (2010–2013), "" (2012–2013), and "Open Heart" (2015).
</item 10>
=== END ITEM id=90f067bf765084af ===

=== ITEM id=d58545f015bbd3c2 ===
Gold sentences and facts:
Sentence 0: Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.
  Fact 0.0: Robin R. Bottin was born on April 1, 1959.
  Fact 0.1: Robin R. Bottin is an American special make-up effects creator.
Sentence 1:  Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing", with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".
  Fact 1.0: Robin R. Bottin is known for collaborations with directors John Carpenter, Paul Verhoeven and David Fincher.
  Fact 1.1: Robin R. Bottin worked with John Carpenter on “The Fog”.
  Fact 1.2: Robin R. Bottin worked with John Carpenter on “The Thing”.
  Fact 1.3: Robin R. Bottin worked with Paul Verhoeven on “RoboCop”.
  Fact 1.4: Robin R. Bottin worked with Paul Verhoeven on “Total Recall”.
  Fact 1.5: Robin R. Bottin worked with Paul Verhoeven on “Basic Instinct”.
  Fact 1.6: Robin R. Bottin worked with David Fincher on “Se7en”.
  Fact 1.7: Robin R. Bottin worked with David Fincher on “Fight Club”.
Sentence 2: David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.
  Fact 2.0: David Andrew Leo Fincher was born on August 28, 1962.
  Fact 2.1: David Andrew Leo Fincher is an American director and producer.
  Fact 2.2: David Andrew Leo Fincher is notably known for films, television series and music videos.
Sentence 3:  He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).
  Fact 3.0: David Fincher was nominated for the Academy Award for Best Director for “The Curious Case of Benjamin Button” (2008).
  Fact 3.1: David Fincher was nominated for the Academy Award for Best Director for “The Social Network” (2010).
Sentence 4:  For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
  Fact 4.0: For “The Social Network”, David Fincher won the Golden Globe Award for Best Director.
  Fact 4.1: For “The Social Network”, David Fincher won the BAFTA Award for Best Direction.

Retrieved items:
<item 1>
Meet the Robinsons
Meet the Robinsons is a 2007 American computer-animated science fiction comedy film produced by Walt Disney Animation Studios and released by Walt Disney
Pictures on March 30, 2007.

 The 47th Disney animated feature film, it was released in standard and Disney Digital 3-D versions.

 The film is loosely based on characters from the children's book "A Day with Wilbur Robinson", by William Joyce.

 The voice cast includes Jordan Fry, Wesley Singerman, Harland Williams, Tom Kenny, Steve Anderson, Laurie Metcalf, Adam West, Tom Selleck, and Angela Bassett.

 It was the first film released after John Lasseter became chief creative officer of Walt Disney Animation Studios.
</item 1>

<item 2>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 2>

<item 3>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 3>

<item 4>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 4>

<item 5>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 5>

<item 6>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 6>

<item 7>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 7>

<item 8>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 8>

<item 9>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 9>

<item 10>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 10>
=== END ITEM id=d58545f015bbd3c2 ===
