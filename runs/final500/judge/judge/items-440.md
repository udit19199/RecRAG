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

This file has 10 items.

=== ITEM id=a821c5711cc16927 ===
Gold sentences and facts:
Sentence 0: "Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series overall.
  Fact 0.0: Moho House is the twenty-first episode of the twenty-eighth season of the animated television series The Simpsons.
  Fact 0.1: Moho House is the 617th episode of The Simpsons overall.
  Fact 0.2: The Simpsons is an animated television series.
Sentence 1:  On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
  Fact 1.0: Fox announced on May 4, 2015.
  Fact 1.1: Fox announced that The Simpsons had been renewed for season 28.

Retrieved items:
<item 1>
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 1>

<item 2>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 2>

<item 3>
The Simpsons (season 28)
The twenty-eighth season of the animated television series "The Simpsons" began airing on Fox in the United States on September 25, 2016, and ended on May 21,
2017.

 On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
</item 3>

<item 4>
The Simpsons Spin-Off Showcase
"The Simpsons Spin-Off Showcase" is the twenty-fourth episode of the eighth season of "The Simpsons".

 It originally aired on the Fox network in the United States on May 11, 1997.

 The episode centers on fictional pilot episodes of non-existent television series derived from "The Simpsons", and is a parody of the tendency of networks to
spin off characters from a hit series.

 As such it includes references to many different TV series.

 The first fictional spin-off is "Chief Wiggum P.I.", a cop-drama featuring Chief Wiggum and Seymour Skinner.

 The second is "The Love-matic Grampa", a sitcom featuring Moe Szyslak who receives dating advice from Abraham Simpson, whose ghost is possessing a love testing
machine.

 The final segment is "The Simpson Family Smile-Time Variety Hour", a variety show featuring the Simpson family except for Lisa, who has been replaced.
</item 4>

<item 5>
Homer to the Max
"Homer to the Max" is the thirteenth episode of "The Simpsons"<nowiki>'</nowiki> tenth season.

 It originally aired on the Fox network in the United States on February 7, 1999.

 In the episode, Homer discovers that a new television show, "Police Cops", has a hero also named Homer Simpson.

 He is delighted with the positive attention he receives because of his name, but when the television character is rewritten from a hero to a bumbling idiot, he
is mocked and taunted, so he changes his name to "Max Power" to rid himself of the negative attention.

 Max gains new friends, and is forced into a protest to prevent a forest from being knocked down.

 In the end, he changes his name back to Homer Simpson.
</item 5>

<item 6>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 6>

<item 7>
South Park (season 10)
The tenth season of "South Park", an American animated television series created by Trey Parker and Matt Stone, began airing on March 22, 2006.

 The tenth season concluded after 14 episodes on November 15, 2006.

 This is the first season in which Kenny does not die and the last season featuring Isaac Hayes (the voice of Chef) as Hayes quit the show following the
backlash behind season nine's "Trapped in the Closet" episode.

 This season also had a minor controversy when the Halloween episode "Hell on Earth 2006" depicted "The Crocodile Hunter's" Steve Irwin with a stingray lodged
in his chest getting thrown out of Satan's Halloween party for not being in costume.

 Episode 2 in this season is the last one with the Braniff Airlines logo.

 All the episodes in this season were written and directed by Trey Parker.
</item 7>

<item 8>
Lisa the Drama Queen
"Lisa the Drama Queen" is the ninth episode of the twentieth season of "The Simpsons".

 It originally aired on the Fox network in the United States on January 25, 2009 and guest starred Emily Blunt as Juliet.

 A special version of the end credits theme was performed by Fall Out Boy, although they do not guest star in the actual episode.

 This was the last episode in the series to be presented only in standard definition, the first regular episode to begin right after the opening credits without
a commercial break, with an episode now having four segments, and the last episode to use the original opening sequence starting from Season 2.

 Brian Kelley returns as a writer after five seasons of being absent from the show.

 The episode is very loosely based on the 1994 film "Heavenly Creatures".
</item 8>

<item 9>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 9>

<item 10>
Judge Me Tender
"Judge Me Tender" is the twenty-third episode and season finale of "The Simpsons'" twenty-first season.

 The 464th episode of the series overall, it originally aired on the Fox network in the United States on May 23, 2010.

 In the episode, Moe discovers his talent for judging in competitions and is invited to appear on the show "American Idol".

 Meanwhile, Homer drives Marge crazy when he starts spending too much time at home, and Lisa tries to comfort Santa's Little Helper.
</item 10>
=== END ITEM id=a821c5711cc16927 ===

=== ITEM id=56c5fadd831ee059 ===
Gold sentences and facts:
Sentence 0: Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".
  Fact 0.0: Muppet Treasure Island is a 1996 American musical adventure comedy film
  Fact 0.1: Muppet Treasure Island is based on Robert Louis Stevenson's Treasure Island
Sentence 1: Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written by Thomas McCarthy.
  Fact 1.0: Million Dollar Arm is a 2014 American biographical sports drama film
  Fact 1.1: Million Dollar Arm was directed by Craig Gillespie
  Fact 1.2: Million Dollar Arm was produced by Walt Disney Pictures
  Fact 1.3: Million Dollar Arm was made from a screenplay written by Thomas McCarthy

Retrieved items:
<item 1>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
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
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 3>

<item 4>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 4>

<item 5>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
</item 5>

<item 6>
Managed Money
Managed Money is a 1934 short comedy film directed by Charles Lamont.

 The film stars Frank Coghlan Jr. and Shirley Temple.

 It was also known as "Frolics of Youth" and "Measured Money".

 This was the second film in which Temple starred as Mary Lou.

 The film tells the story of Sonny and Sid prospecting for gold to pay for their military academy education.

 Sonny's sister, Mary Lou stows away in the back of their car.

 Eventually, they meet an inventor who helps them financially.
</item 6>

<item 7>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 7>

<item 8>
Rinku Singh
Rinku Singh (born August 8, 1988 in Lucknow, India) is a left-handed baseball pitcher who is currently a free agent.

 Singh was signed by the Pittsburgh Pirates organization after he won a pitching contest on a 2008 reality television show, "The Million Dollar Arm".

 He was the first Indian to play professional baseball and has spent several seasons in the minor leagues, only reaching as high as the A level.

 He is the subject of the movie "Million Dollar Arm".
</item 8>

<item 9>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 9>

<item 10>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 10>
=== END ITEM id=56c5fadd831ee059 ===

=== ITEM id=83febe58d8a2406d ===
Gold sentences and facts:
Sentence 0: Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.
  Fact 0.0: Emmitt H. Ford was born on December 13, 1943.
  Fact 0.1: Emmitt H. Ford died on November 10, 2014.
  Fact 0.2: Emmitt H. Ford was an American politician.
  Fact 0.3: Emmitt H. Ford was from Tennessee.
Sentence 1:  He was also an uncle of former United States Congressman Harold Ford Jr.
  Fact 1.0: Emmitt H. Ford was an uncle of Harold Ford Jr.
  Fact 1.1: Harold Ford Jr. is a former United States Congressman.
Sentence 2:  He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.
  Fact 2.0: Emmitt H. Ford is the son of Harold Ford Sr.
  Fact 2.1: Harold Ford Sr. is a former Congressman.
  Fact 2.2: Harold Ford Sr. held the same seat for 22 years.

Retrieved items:
<item 1>
John Anderson (Maine politician)
John Anderson (1792–1853) was a Maine politician.

 Anderson served as United States Representative from Maine from 1825-1833.
</item 1>

<item 2>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 2>

<item 3>
L. R. Ford Jr.
Lester Randolph Ford Jr. (born September 23, 1927 – February 26, 2017) was an American mathematician specializing in network flow problems.

 He was the son of mathematician Lester R. Ford Sr.
</item 3>

<item 4>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 4>

<item 5>
Harold Ford Jr.
Harold Eugene Ford Jr. (born May 11, 1970) is an American politician and author who served five terms in the United States House of Representatives as a member
of the Democratic Party from  's 9 congressional district , centered in Memphis, from 1997 to 2007.

 He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.

 In 2006, Ford made an unsuccessful bid for the US Senate seat vacated by retiring Bill Frist.

 He is a member of the Ford political family from Memphis.

 Ford was the last chairman of the Democratic Leadership Council (DLC).
</item 5>

<item 6>
John Ford (Tennessee politician)
John N. Ford (born May 3, 1942 in Memphis, Tennessee), is a former Democratic member of the Tennessee State Senate and a member of Tennessee's most prominent
African-American political family.

 He is the older brother of former U.S. Representative Harold Ford, Sr. and the uncle of former Tennessee U.S. Representative and 2006 United States Senate
candidate Harold Ford, Jr. He is a member of the Ford political family from Memphis.
</item 6>

<item 7>
Emmitt Ford
Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.

 He represented the 86th district encompassing Shelby County from 1975–1981.

 He was also an uncle of former United States Congressman Harold Ford Jr.
</item 7>

<item 8>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 8>

<item 9>
United States Senate election in Tennessee, 2006
The 2006 United States Senate election in Tennessee was held on November 7, 2006.

 The election winner, Bob Corker, served his term between January 3, 2007 and January 3, 2013.

 Corker replaced Republican Bill Frist in the Senate, who retired upon the end of his second term in 2007.

 Corker was the Republican nominee, and the Democratic nominee was Harold Ford, Jr..

 The race between Ford and Corker was one of the most competitive Senate races of 2006, with Corker winning the race by less than three percent of the vote.

 Corker was the only non-incumbent Republican to win a U.S. Senate seat in 2006.

 Since 1994, the Republican Party has held both of Tennessee's Senate seats.
</item 9>

<item 10>
Doug Ford Jr.
Douglas Bruce Ford Jr. (born November 20, 1964) is a Canadian businessman and politician in Toronto, Ontario.

 Ford was Toronto City Councillor for Ward 2 Etobicoke North in Toronto from 2010 to 2014 at the same time that his brother, Rob Ford (1969–2016), was mayor of
Toronto.

 Their father, Doug Ford Sr., was a Member of Provincial Parliament from 1995 to 1999 and founded Deco Labels and Tags, a printing business operating in Canada
and the United States.

 The business is now owned by Doug Ford Jr., who has served as company president since 2002, and his brother, Randy.
</item 10>
=== END ITEM id=83febe58d8a2406d ===

=== ITEM id=2e61f81419c3ec98 ===
Gold sentences and facts:
Sentence 0: The Nashville Tribute Band is a Mormon music group founded by Jason Deere and Dan Truman, the pianist of the popular country group Diamond Rio.
  Fact 0.0: The Nashville Tribute Band is a Mormon music group.
  Fact 0.1: The Nashville Tribute Band was founded by Jason Deere.
  Fact 0.2: The Nashville Tribute Band was founded by Dan Truman.
  Fact 0.3: Dan Truman is the pianist of the popular country group Diamond Rio.
Sentence 1: Diamond Rio is an American country and Christian country music band.
  Fact 1.0: Diamond Rio is an American country music band.
  Fact 1.1: Diamond Rio is a Christian country music band.
Sentence 2:  The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then the Tennessee River Boys.
  Fact 2.0: Diamond Rio was founded in 1982.
  Fact 2.1: Diamond Rio was founded as an attraction for the Opryland USA theme park in Nashville, Tennessee.
  Fact 2.2: Diamond Rio was originally known as the Grizzly River Boys.
  Fact 2.3: Diamond Rio was later known as the Tennessee River Boys.

Retrieved items:
<item 1>
The entries profile rock, punk, and metal albums, bands, and musicians. Fuel’s *Puppet Strings* (2014) reunited singer Brett Scallions but featured none of the
prior album’s recording personnel. Jizzy Pearl sang on Quiet Riot’s *10* and L.A. Guns’ *Shrinking Violet*. Other subjects include J. Geils Band, Green Day side
projects and collaborators, Glassjaw and Daryl Palumbo, Clearview 77, Sons of Abraham,
</item 1>

<item 2>
The Pilgrims (band)
The Pilgrims are an alternative rock band from Windsor, Vermont.

 Seven Days contributor Dan Bolles compared their style to alt-punk and indie rock, while John Powell, writing for www.angelica-music.com, has likened them to
rock and roll bands from the 1960s and 1970s.

 They are part of the Windsor, VT music collective What Doth Life, which has released albums by The Pilgrims, Giant Travel Avant Garde, Derek and The Demons,
Carton, and Luke Chrisinger.
</item 2>

<item 3>
Grizzly River Run
Grizzly River Run is located at Disney California Adventure at the Disneyland Resort in Anaheim, California.

 It is similar to Kali River Rapids in Disney's Animal Kingdom but distinctive as the rafts are engineered to spin as they descend chutes.

 The attraction's name comes from Grizzly Peak, the bear shaped mountain that the rapids flow around.

 It was designed by Walt Disney Imagineering and constructed by Intamin.
</item 3>

<item 4>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 4>

<item 5>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 5>

<item 6>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 6>

<item 7>
The Studdogs
The Studdogs was a rock band from Orlando, Florida.

 The band's music contains elements of punk, blues rock, garage rock, and noise rock.

 The band was formed in 2000 and stayed active until 2005.

 They released a lot of material, particularly the full-length album, "The Gospel According to the Studdogs" and the 7" vinyl 3-song EP, and gained local and
national notoriety.

 They were featured in several national music magazines, including Magnet and Amplifier.

 Comparisons were made to the Stooges, Dead Boys, and it was said they sounded "like absolute victory by way of slurring, Rolling Stones-y garage rock from
nowhere in particular."

 Their drunken, abrasive live shows often ended with someone bleeding or being thrown out of the venue.

 Their live shows were said to be "what Howling Wolf would sound like through a wall of noise and distortion."
</item 7>

<item 8>
The Hillmen (album)
The Hillmen is a studio album by The Hillmen, a southern California bluegrass band originally known as The Golden State Boys.

 The Hillmen consisted of Chris Hillman (later of The Byrds, The Flying Burrito Brothers, Manassas and The Desert Rose Band) on mandolin, country
singer/songwriter Vern Gosdin on guitar and lead vocals, his brother Rex Gosdin on double bass, and Don Parmley (later of the Bluegrass Cardinals) on banjo.
</item 8>

<item 9>
The material covers country, country-rock, folk-rock and bluegrass artists and releases. Oak Ridge Boys’ 2009 *The Boys Are Back* marked a return to secular
country, led by a White Stripes cover. Richie Furay helped form Buffalo Springfield and Poco, later joining the Souther–Hillman–Furay Band; Poco’s debut
featured his “Pickin’ Up the Pieces.” Other subjects include Buffalo Springfield’s final album, the Lonesome Pine F
</item 9>

<item 10>
Nashville Tribute Band
The Nashville Tribute Band is a Mormon music group founded by Jason Deere and Dan Truman, the pianist of the popular country group Diamond Rio.
</item 10>
=== END ITEM id=2e61f81419c3ec98 ===

=== ITEM id=aaff9cb0b7507393 ===
Gold sentences and facts:
Sentence 0:  For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record when her album "25" took just 29 days to reach the landmark.
  Fact 0.0: For almost 20 years, a record (held by a subject not named in this sentence) stood for selling 2 million copies in the UK in the fastest time.
  Fact 0.1: The record for selling 2 million copies in the UK in the fastest time was set in 42 days.
  Fact 0.2: In 2015, Adele surpassed the record.
  Fact 0.3: Adele's album '25' took just 29 days to reach the 2 million copies landmark.
Sentence 1:  At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
  Fact 1.0: The 51st Grammy Awards were held in 2009.
  Fact 1.1: Adele received the Best New Artist award at the 51st Grammy Awards.
  Fact 1.2: Adele received the Best Female Pop Vocal Performance award at the 51st Grammy Awards.

Retrieved items:
<item 1>
List of Billboard 200 number-one albums of 2002
Country music singer Shania Twain's album "Up!

" opened up with a huge first week sales of 857,000 copies in its first week alone, giving her the recognition of the highest first week sales of her career and
second highest of the year, only behind Eminem's "The Eminem Show" and at the time the fastest selling solo female album ever.

 Nelly's album "Nellyville" opened up with his highest first week sales of his career which logged on with huge sales of 714,000 copies in its first week alone,
which beat his sales of his debut album "Country Grammar", which opened up with first week sales of 235,000 copies.

 Country singer Alan Jackson album "Drive" gave him his first number one album on the chart and opened up with first week sales of 211,000 copies alone.
</item 1>

<item 2>
List of Billboard 200 number-one albums of 2002
The band Creed continued its eight week long run on the chart but is credited as the longest running album 2001.

 Jennifer Lopez earned her second number one album on the charts with "", which became the highest first week sales of a remix album at the time.

 R&B artist Ashanti earned her first number one album with her self-titled debut album "Ashanti", which opened up with first week sales of 503,000 copies in its
first week alone.

 Puff Daddy earned his first number one album since "No Way Out" back in 1997.

 Rapper Jay-Z earned his fifth chart topper with "", which opened up with first week sales of 545,000 copies alone.

 Heavy metal band Disturbed earned its first number one album on the chart with "Believe", which opened up with first week sales of 284,000 copies alone.

 Country music singer Shania Twain's album "Up!

"
</item 2>

<item 3>
MTV Video Music Award for Best New Artist
The MTV Video Music Award for Best New Artist have been given out since the first annual MTV Video Music Awards in 1984.

 From then to 2006, the award was named Best New Artist in a Video.

 In 2007 its name was changed to Best New Artist, as the category underwent a format change to award the artist's body of work for the full year rather than a
specific video.

 For the 2008 ceremony, though, while the award retained its 2007 name, it returned to the format of awarding a specific video rather than the artist's full
body of work.

 The category was later renamed Artist to Watch through 2013 to 2015 while still keeping the format of an award going to a certain video.
</item 3>

<item 4>
Billboard Year-End
"Billboard" Year-End charts are a cumulative measure of a single or album's performance in the United States, based upon the "Billboard" magazine charts during
any given chart year.

 "Billboard's" "chart year" runs from the first Billboard "week" of December to the final week in November, but because the Billboard week is dated in advance
of publication, the last calendar week for which sales are counted is usually the third week in November.

 This altered calendar allows for "Billboard" to calculate year-end charts and release them in time for its final print issue in the last week of December.

 Prior to incorporating chart data from Nielsen SoundScan (from 1991), year-end charts were calculated by an inverse-point system based solely on a title's
performance (for example a single appearing on the "Billboard" Hot 100 would be given one point for a week spent at position 100, two points for a week spent at
position ninety-nine, and so forth, up to 100 points for each week spent at number one).

 Other factors including the total weeks a song spent on the chart and at its peak position were calculated into its year-end total.
</item 4>

<item 5>
Japan Record Award for Best New Artist
The Japan Record Award for Best New Artist (最優秀新人賞 ) is awarded annually.

 Until the 10th Japan Record Awards it was called New Artist Award, since 11th — Best New Artist Award.

 At present all nominees for the Best New Artist Award are awarded the New Artist Award.
</item 5>

<item 6>
List of accolades received by Grand Theft Auto V
Within twenty-four hours of its release, "Grand Theft Auto V" generated more than $815.7 million in worldwide revenue, equating to approximately 11.21 million
copies sold for Take Two.

 Three days after release, the game had surpassed one billion dollars in sales, making it the fastest selling entertainment product in history.
</item 6>

<item 7>
List of Billboard 200 number-one albums of 2002
The highest-selling albums and EPs in the United States are ranked in the "Billboard" 200, published by "Billboard" magazine.

 The data are compiled by Nielsen Soundscan based on each album's weekly physical and digital sales.

 25 acts achieved number one albums during this year with artist such as Nelly and Shania Twain who had their albums debut at number one on the chart.

 Rapper Eminem's "The Eminem Show" is the best selling album of 2002 selling over approximately 7.6 million copies by the end of the year.

 It is also the longest running album of 2002 spending six non-consecutive weeks the chart and was known for its first full week of sales debut of 1.322 million
copies which Nielsen SoundScan scanned as the sixth largest sales of all time in its first week.

 Its debut of 1.322 million copies has still not been matched by any album today since except for Taylor Swift's album "1989", which opened with first week
sales of 1.279 million copies.
</item 7>

<item 8>
Robson &amp; Jerome (album)
Robson & Jerome is the debut studio album by British singing duo Robson & Jerome.

 It was released in the UK in 1995 by RCA Records, and reached number one on the UK Albums Chart.

 It was the Christmas number one album of 1995, and was the biggest-selling album of the year, with sales of 2,040,000.

 For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record
when her album "25" took just 29 days to reach the landmark.
</item 8>

<item 9>
Adele
Adele Laurie Blue Adkins (born 5 May 1988) ( ) is an English singer-songwriter.

 After graduating from the BRIT School for Performing Arts and Technology in 2006, Adele was given a recording contract by XL Recordings after a friend posted
her demo on Myspace the same year.

 In 2007, she received the Brit Awards "Critics' Choice" award and won the BBC Sound of 2008 poll.

 Her debut album, "19", was released in 2008 to commercial and critical success.

 It is certified seven times platinum in the UK, and three times platinum in the US.

 The album contains her first song, "Hometown Glory", written when she was 16, which is based on her home suburb of West Norwood in London.

 An appearance she made on "Saturday Night Live" in late 2008 boosted her career in the US.

 At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
</item 9>

<item 10>
Billboard Year-End
The same method was used for albums based on the "Billboard" 200, and songs appearing on the other charts (e.g. Hot Country Singles).
</item 10>
=== END ITEM id=aaff9cb0b7507393 ===

=== ITEM id=215a25646c2bbe14 ===
Gold sentences and facts:
Sentence 0: Pravda (Russian: Правда ; ] , "Truth") is a Russian broadsheet newspaper, formerly the official newspaper of the Communist Party of the Soviet Union, when it was one of the most influential papers in the country with a circulation of 11 million.
  Fact 0.0: Pravda's Russian name is Правда, which means 'Truth'.
  Fact 0.1: Pravda is a Russian broadsheet newspaper.
  Fact 0.2: Pravda was formerly the official newspaper of the Communist Party of the Soviet Union.
  Fact 0.3: When Pravda was the official newspaper of the Communist Party of the Soviet Union, it was one of the most influential papers in the country.
  Fact 0.4: When Pravda was one of the most influential papers in the country, its circulation was 11 million.
Sentence 1: Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.
  Fact 1.0: Stiffelio is an opera.
  Fact 1.1: Stiffelio is an opera in three acts.
  Fact 1.2: Stiffelio is by Giuseppe Verdi.
  Fact 1.3: Stiffelio uses an Italian libretto by Francesco Maria Piave.

Retrieved items:
<item 1>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 1>

<item 2>
Graph query row:
work: Stiffelio
kind: opera
genre: None
relationship: None
genre_entity: None
</item 2>

<item 3>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 3>

<item 4>
Pravda Pyat
Pravda Pyat (Russian: правда пять or "Truth Five") is a weekly Russian tabloid news publication that was a spin-off from "Pravda".

 It was founded by Greek entrepreneurs Christos Giannikos and Fyodoros Giannikos of Pravda International in 1996.

 The magazine replaced "Pravda" when that publication ceased operations.

 Its intended audience was younger readers, and coverage was more sensationalistic than "Pravda", focusing on crime and scandals.
</item 4>

<item 5>
Pravda
Pravda (Russian: Правда ; ] , "Truth") is a Russian broadsheet newspaper, formerly the official newspaper of the Communist Party of the Soviet Union, when it
was one of the most influential papers in the country with a circulation of 11 million.
</item 5>

<item 6>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 6>

<item 7>
Steamer Pravda
Steamer "Pravda" was a Soviet merchant freighter of about 3,100 tonnes displacement, which was active in the Soviet Arctic during the 1930s.

 This ship had been normally used for carrying timber.

 It was named after Soviet newspaper "Pravda".
</item 7>

<item 8>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 8>

<item 9>
Graph query row:
work: Pravda
kind: Soviet newspaper
genre: None
relationship: None
genre_entity: None
</item 9>

<item 10>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 10>
=== END ITEM id=215a25646c2bbe14 ===

=== ITEM id=27e32faccff19c07 ===
Gold sentences and facts:
Sentence 0: Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).
  Fact 0.0: Groom Lake is a salt flat in Nevada.
  Fact 0.1: Groom Lake is used for runways of the Nellis Bombing Range Test Site airport.
  Fact 0.2: The Nellis Bombing Range Test Site airport has the code KXTA.
Sentence 1:  Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
  Fact 1.0: Groom Lake is located within the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.1: Groom Lake Valley is the namesake of the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.2: Groom Lake is 25 miles south of Rachel, Nevada.
Sentence 2: Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.
  Fact 2.0: Rachel is a census-designated place (CDP).
  Fact 2.1: Rachel is in Lincoln County, Nevada, United States.

Retrieved items:
<item 1>
Papoose Lake
Papoose Lake is a dry lake bed located in Lincoln County, Nevada, USA.

 The lake lies within the plot of land referred to as the Groom Lake facility (aka Area 51) and is a restricted area.

 The lake is located a few miles southwest of the Groom lake facility nestled adjacent the Papoose Range.
</item 1>

<item 2>
Saskatchewan Highway 35
Highway 35 is a paved undivided provincial highway in the Canadian province of Saskatchewan.

 It runs from the US Border near Port of Oungre (where it meets United States Route 85) to a dead end near the north shore of Tobin Lake.

 Saskatchewan Highway 35 is about 540 km long.

 The CanAm Highway comprises Saskatchewan Highways 35, SK Hwy 39, SK Hwy 6, SK Hwy 3, SK Hwy 2 and U.S. Route 85.

 46.5 mi of SK Hwy 35 contribute to the CanAm Highway between Port of Oungre on the Canada – United States border and Weyburn.

 Mudslides, and spring flooding were huge road building and maintenance problems around Nipawin as well as along the southern portion of the route named the
"Greater Yellow Grass Marsh".

 Over 20 early dams were built until the problem was addressed with the Rafferty-Alameda Project on the Souris River and the construction of the Qu'Appelle
River Dam which have helped to eliminate washed out roads and flooded communities.
</item 2>

<item 3>
Squire Creek Country Club
The Squire Creek Country Club is a private, members-only country club located in Choudrant, Louisiana, five miles northeast of Ruston.

 Squire Creek features an 18-hole championship golf course designed by Tom Fazio.

 The course has been ranked as number one in the state of Louisiana four times, and was ranked as the No. 5 best new course in the United States by "Golf
Digest".

 Squire Creek is the home golf course for the Louisiana Tech Bulldogs golf team.

 Squire Creek offers a golf training facility, which features dual bays with retractable doors for indoor or outdoor use, motion analysis, and launch monitor.

 The Squire Creek tennis facility has 6 lighted courts (4 Hydro Courts and 2 Hard Courts).

 Squire Creek has full service golf and tennis shops.

 The Squire Creek fitness facility has 2 exercise rooms and offers massage therapy.

 The Squire Creek Clubhouse features three dining areas: the more formal Main Dining Hall, the casual 19th Hole, and the Fazio Grill.

 The Squire Creek Lodge features two floors offering a combined total of seven bedrooms.
</item 3>

<item 4>
Rachel, Nevada
Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.

 As of the 2010 census it had a population of 54.

 As the closest habitation to the Nellis Air Force Range and Area 51, Rachel enjoys a modest celebrity, particularly among aviation enthusiasts and UFO hunters.

 North of the town is the Quinn Canyon Range, which has the ghost town of Adaven.
</item 4>

<item 5>
Leviathan Mine
Leviathan Mine is a United States superfund site (CERCLIS ID: CAD98067685) at an abandoned open-pit sulfur mine located in Alpine County, California.

 The mine is located at on the eastern slope of the Sierra Nevada at about 7,000 ft elevation, 6 mi east of Markleeville and 24 mi southeast of Lake Tahoe.

 The mine site comprises approximately 250 acre of land surrounded by the Toiyabe National Forest, which is only accessible a few months a year.

 The approximately 22 million tons of sulfur ore-containing crushed rock at the mine are responsible for contaminating the Leviathan and Aspen Creek, which join
with Mountaineer Creek to form Bryant Creek which ultimately empties into the East Fork of the Carson River.

 These water bodies are listed as 303(d) impaired.

 The site location is seismically active.
</item 5>

<item 6>
Crystal Lake (Gilmanton, New Hampshire)
Crystal Lake is a 455 acre water body located in Belknap County in the Lakes Region of central New Hampshire, United States, in the town of Gilmanton.

 Crystal Lake is at the head of the Suncook River watershed.

 Water flows into Crystal Lake from the Belknap Range through Manning and Sunset lakes to the north.

 Water from Manning Lake enters Crystal Lake via Nelson Brook.

 Other waterways flowing into Crystal Lake include Wasson Brook and Mill Brook.

 Water flows out of a dam from the southern end of Crystal Lake via the Suncook Lakes and Suncook River to the Merrimack River.

 Shoreline development along Crystal Lake consists primarily of summer cottages, with a few year-round residents.

 Belknap Mountain and Mount Major can be seen from the shoreline.
</item 6>

<item 7>
Groom Lake (salt flat)
Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).

 It is located immediately north of the Area 51 USAF installation.

 The lake at 4409 ft elevation is approximately 3.7 mi from north to south and 3 mi from east to west at its widest point, and is approximately 11.3 miles in
circumference.

 Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
</item 7>

<item 8>
John Skillern House
The John Skillern House is a historic cabin located 25 mi northwest of Fairfield in Camas County, Idaho, near the confluence of the Big Smokey and Little Smokey
creeks.

 The cabin was built in 1921-22 for John Skillern and his wife, who used it as a summer home and headquarters for John's large sheep ranching business.

 Skillern's wife based the cabin's rustic design off of the Old Faithful Inn in Yellowstone National Park, which is reflected in the cabin's steep roof and
covered front porch.

 The cabin's other significant rustic features include its sleeping lofts with pole railings, its horizontal log construction with exposed logs on the inner
walls, and its stone chimney.
</item 8>

<item 9>
Groom Range
The Groom Range is a mountain range in Lincoln County, Nevada.

 It is located within the Nevada Test and Training Range, north of Groom Dry Lake.

 The highest point in the Groom Range is 9,249 feet.

 The Groom Range is situated 26.6 miles north of the dry Groom Lake.
</item 9>

<item 10>
Tonopah Test Range
Tonopah Test Range (TTR), is a restricted military installation located about 30 mi southeast of Tonopah, Nevada.

 It is part of the northern fringe of the Nellis Range, measuring 625 sqmi .

 Tonopah Test Range is located about 70 mi northwest of Groom Dry Lake, the home of the Area 51 facility.

 Like the Groom Lake facility, Tonopah is a site of interest to conspiracy theorists, mostly for its use of experimental and classified aircraft.

 As such, it is not the focus of alien enthusiasts, unlike its neighbor.

 It is currently used for nuclear weapons stockpile reliability testing, research and development of fusing and firing systems, and testing nuclear weapon
delivery systems.

 The airspace comprises restricted area R-4809 of the Nevada Test and Training Range and is often used for military training.
</item 10>
=== END ITEM id=27e32faccff19c07 ===

=== ITEM id=09a25f962193036c ===
Gold sentences and facts:
Sentence 0: Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.
  Fact 0.0: Paola Suárez was born on 23 June 1976.
  Fact 0.1: Paola Suárez is a retired professional tennis player.
  Fact 0.2: Paola Suárez is from Argentina.
Sentence 1:  She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.
  Fact 1.0: Paola Suárez was one of the most prominent women's doubles players throughout the early and mid-2000s.
  Fact 1.1: Paola Suárez won eight Grand Slam titles.
  Fact 1.2: All eight of Paola Suárez's Grand Slam titles were won with Virginia Ruano Pascual.
  Fact 1.3: Paola Suárez held the No. 1 doubles ranking for 87 non-consecutive weeks.
Sentence 2:  He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.
  Fact 2.0: Paola Suárez reached a career high singles ranking of 274 in July 2011.
  Fact 2.1: Paola Suárez is much better known as a leading doubles player with a huge serve.

Retrieved items:
<item 1>
1999 Brasil Open – Doubles
The 1999 Brasil Open – Doubles was the tennis doubles event of the first edition of the most prestigious tournament in Brazil.

 Argentinian team Laura Montalvo and Paola Suárez won the title, in what was the team's second title of the year, defeating Janette Husárová and Florencia Labat
in the final.
</item 1>

<item 2>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 2>

<item 3>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 3>

<item 4>
1999 Copa Colsanitas – Singles
The 1999 Copa Colsanitas – Singles was the singles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Paola Suárez was the defending champion but lost in the semifinals to Christína Papadáki.
</item 4>

<item 5>
Patricia Tarabini
Patricia Tarabini (born 6 August 1968) is an Argentine former tennis player.

 Tarabini represented her country and won the bronze Olympic medal at the 2004 Athens games, with Paola Suárez, where they lost 7–9 in the third set to eventual
gold medalists from China, Sun Tiantian and Li Ting.

 On May 9, 1988, Patricia reached her highest singles ranking; world number 29.

 Tarabini's highest doubles ranking was world number 12, which she achieved on August 17, 1998.

 Patricia turned pro in 1986, and won a total of 15 top-level doubles titles in her career.

 She is the 1996 French Open mixed doubles champion, which she won with Javier Frana.
</item 5>

<item 6>
Clarisa Fernández
Clarisa Fernández (born 28 August 1981) is a retired Argentine tennis player who is best known for her semi-final appearance at the 2002 French Open, the first
Argentine since Sabatini in 1992 to reach that stage.

 Fernández was ranked 87th in the world at the time of her shocking result.

 She defeated, in order: (Q) Lubomira Bacheva, Jelena Kostanić, Kim Clijsters, (13) Elena Dementieva, Paola Suárez, before losing to second-seeded Venus
Williams in straight sets.
</item 6>

<item 7>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 7>

<item 8>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 8>

<item 9>
1999 Italian Open – Women's Doubles
The 1999 Italian Open – Doubles was the doubles event of the fifty-fifth edition of the tennis tournament played at Rome, Italy, the most prestigious tennis
tournament in Southern Europe.

 It was the fifth WTA Tier I tournament of the year, and part of the European claycourt season.

 Virginia Ruano Pascual and Paola Suárez were the defending champions, but lost to Martina Hingis and Anna Kournikova in the quarterfinals.
</item 9>

<item 10>
Paola Suárez
Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.

 She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano
Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.

 She was also a singles semifinalist at the 2004 French Open.
</item 10>
=== END ITEM id=09a25f962193036c ===

=== ITEM id=62a324accd33ae7d ===
Gold sentences and facts:
Sentence 0: of Montreal is an American experimental pop band from Athens, Georgia.
  Fact 0.0: of Montreal is an American band.
  Fact 0.1: of Montreal is an experimental pop band.
  Fact 0.2: of Montreal is from Athens, Georgia.
Sentence 1: Vib Gyor was a band that was formed in Leeds, England.
  Fact 1.0: Vib Gyor was a band.
  Fact 1.1: Vib Gyor was formed in Leeds, England.

Retrieved items:
<item 1>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 1>

<item 2>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 2>

<item 3>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 3>

<item 4>
Bloc Party
Bloc Party are an English indie rock band, currently composed of Kele Okereke (lead vocals, rhythm guitar, keyboards, sampler), Russell Lissack (lead guitar,
keyboards), Justin Harris (bass guitar, keyboards, saxophones, backing vocals) and Louise Bartle (drums, percussion).

 Former members Matt Tong and Gordon Moakes left the band in 2013 and 2015 respectively.

 Their brand of music, whilst rooted in rock, retains elements of other genres such as electronica and house music.

 The band was formed at the 1999 Reading Festival by Okereke and Lissack.

 They went through a variety of names before settling on Bloc Party in 2003.

 Moakes joined the band after answering an advert in "NME" magazine, while Tong was picked via an audition.

 Bloc Party got their break by giving BBC Radio 1 DJ Steve Lamacq and Franz Ferdinand's lead singer, Alex Kapranos, a copy of their demo "She's Hearing Voices".
</item 4>

<item 5>
The Famines (band)
The Famines are a Canadian indie rock band formed in 2008 in Edmonton, Alberta, Canada now based in Montreal, Quebec, Canada.

 The two piece band uses a modern and minimalistic approach that draws comparison to mid 1970's protopunk and fuzzy garage rock.

The band name is meant to be a commentary on the continued feeling of emptiness and lacking in a society that is materially fulfilled.

 The band has two members, R. E. Biesinger on guitar and vocals, and Drew Demers on the drums.
</item 5>

<item 6>
The Gorgeous
The Gorgeous were a Canadian metalcore band formed in the summer of 2003 in Montreal.

 The band formed with Dali Shaw on guitar, Julien Brousseau on Bass and Miguel Shaw on drums.

 The three played a few shows without a vocalist and quickly recruited Jordon Daniel.
</item 6>

<item 7>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 7>

<item 8>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
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
Fallen (Vib Gyor song)
"Fallen" was the first single from the Leeds band Vib Gyor.

 It was first given limited release in July 2006, and has attained critical acclaim and a large amount of radio play on both sides of the atlantic.
</item 10>
=== END ITEM id=62a324accd33ae7d ===

=== ITEM id=de1f9d091de30165 ===
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
Boeing F/A-18E/F Super Hornet
The Boeing F/A-18E and F/A-18F Super Hornet are twin-engine carrier-capable multirole fighter aircraft variants based on the McDonnell Douglas F/A-18 Hornet.

 The F/A-18E single-seat and F/A-18F tandem-seat variants are larger and more advanced derivatives of the F/A-18C and D Hornet.

 The Super Hornet has an internal 20 mm M61 rotary cannon and can carry air-to-air missiles and air-to-surface weapons.

 Additional fuel can be carried in up to five external fuel tanks and the aircraft can be configured as an airborne tanker by adding an external air refueling
system.
</item 1>

<item 2>
No. 1 Squadron RAAF
No. 1 Squadron is a Royal Australian Air Force (RAAF) squadron headquartered at RAAF Base Amberley, Queensland.

 Controlled by No. 82 Wing, it is equipped with Boeing F/A-18F Super Hornet multi-role fighters.

 The squadron was formed under the Australian Flying Corps in 1916 and saw action in the Sinai and Palestine Campaigns during World War I.

 It flew obsolete Royal Aircraft Factory B.E.2s, B.E.12s, Martinsyde G.100s and G.102s, as well as Airco DH.6s, Bristol Scouts and Nieuport 17s, before
re-equipping with the R.E.8 in October 1917 and finally the Bristol Fighter in December.

 Its commanding officer in 1917–18 was Major Richard Williams, later known as the "Father of the RAAF".

 Disbanded in 1919, No. 1 Squadron was re-formed on paper as part of the RAAF in 1922, and re-established as an operational unit three years later.
</item 2>

<item 3>
Boeing EA-18G Growler
The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.

 The EA-18G replaced the Northrop Grumman EA-6B Prowlers in service with the United States Navy.

 The Growler's electronic warfare capability is primarily provided by Northrop Grumman.

 The EA-18G began production in 2007 and entered operational service with the US Navy in late 2009.

 Australia has also purchased twelve EA-18Gs, which entered service with the Royal Australian Air Force in 2017.
</item 3>

<item 4>
VAQ-209
Electronic Attack Squadron 209 (VAQ-209) is a United States Navy Reserve electronic attack squadron.

 Known as the "Star Warriors", the squadron flies the EA-18G Growler carrier-based electronic warfare jet aircraft.

 Based at Naval Air Station Whidbey Island, WA, it is assigned to the Tactical Support Wing.
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
No. 82 Wing RAAF
No. 82 Wing is the strike and reconnaissance wing of the Royal Australian Air Force (RAAF).

 It is headquartered at RAAF Base Amberley, Queensland.

 Coming under the control of Air Combat Group, the wing operates F/A-18F Super Hornet multirole fighters and Pilatus PC-9 forward air control aircraft.

 Its units include Nos. 1 and 6 Squadrons, operating the Super Hornet, and No. 4 Squadron, operating the PC-9.
</item 6>

<item 7>
RAAF Base Amberley
RAAF Base Amberley (ICAO: YAMB) is a Royal Australian Air Force (RAAF) military airbase located 8 km southwest of Ipswich, Queensland in Australia and 50 km
southwest of Brisbane.

 It is currently home to No. 1 Squadron and No. 6 Squadron (operating the F/A-18F Super Hornet), No. 33 Squadron (taking delivery of the Airbus KC-30A) and No.
36 Squadron (operating the Boeing C-17 Globemaster III jet transport).

 Amberley is also home to Army units making up the 9th Force Support Battalion (9FSB).

 Located on 1600 ha , RAAF Amberley is the largest operational base in the RAAF, employing over 5,000 uniformed and civilian personnel.

 There are a variety of other formations on the base such as training colleges and maintenance areas.

 Amberley's largest squadron in terms of personnel is No. 382 Expeditionary Combat Support Squadron RAAF (ECSS) providing both garrison and deployed combat
support.
</item 7>

<item 8>
VAQ-138
Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at
Naval Air Station Whidbey Island, Washington.

 Their radio callsign is "RAMPAGE".
</item 8>

<item 9>
VFA-154
Strike Fighter Squadron 154 (VFA-154), also known as the "Black Knights", is a United States Navy strike fighter squadron stationed at Naval Air Station
Lemoore.

 The Black Knights are an operational fleet squadron flying the F/A-18F Super Hornet.

 They are currently attached to Carrier Air Wing Eleven and deployed aboard the aircraft carrier USS "Nimitz" .

 Their tailcode is NH and their callsign is "Knight".
</item 9>

<item 10>
VAQ-135
Electronic Attack Squadron 135 (VAQ-135), known as the "Black Ravens", is a United States Navy electronic attack squadron that currently operates the EA-18G
Growler carrier-based electronic warfare jet aircraft.

 The squadron is permanently stationed at Naval Air Station Whidbey Island with a radio callsign of ""Thunder"".
</item 10>
=== END ITEM id=de1f9d091de30165 ===
