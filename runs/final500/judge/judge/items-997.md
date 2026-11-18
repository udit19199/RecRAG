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

This file has 7 items.

=== ITEM id=8c06e1c343d30d6b ===
Gold sentences and facts:
Sentence 0: The film is based loosely on Eminem 's actual upbringing , and follows white rapper B - Rabbit ( Eminem ) and his attempt to launch a career in a genre dominated by African - Americans . The title is derived from 8 Mile Road , the highway between the predominantly black city of Detroit and Wayne County and the predominantly White Oakland County and Macomb County suburbs . It was filmed mostly on location .
  Fact 0.0: The film is based loosely on Eminem's actual upbringing.
  Fact 0.1: The film follows white rapper B-Rabbit (Eminem).
  Fact 0.2: B-Rabbit (Eminem) attempts to launch a career in a genre dominated by African-Americans.
  Fact 0.3: The title of the film is derived from 8 Mile Road.
  Fact 0.4: 8 Mile Road is the highway between the predominantly black city of Detroit and Wayne County.
  Fact 0.5: 8 Mile Road is the highway between the predominantly White Oakland County and Macomb County suburbs.
  Fact 0.6: The film was filmed mostly on location.

Retrieved items:
<item 1>
8 Mile (film)
Chicago Sun - Times . Retrieved 2011 - 02 - 27 . Jump up ^ http://apps.tvplex.go.com/ebertandthemovies/audioplayer.cgi?file=021111_8_mile Jump up ^ `` 8 Mile ''
. Rolling Stone . Retrieved 15 September 2016 . Jump up ^ Ramirez , Erika ( November 8 , 2012 ) . `` Top 10 Best Hip - Hop Movies Ever '' . Billboard .
Prometheus Global Media . Retrieved February 17 , 2014 . Jump up ^ Sarris , Andrew ( January 13 , 2003 ) . `` The Best Films of 2002 , And a Few Honorable
Mentions '' . The New York Observer . The New York Observer , LLC . Retrieved February 17 , 2014 . Jump up ^ Schickel , Richard ( December 12 , 2002 ) . `` Top
10 Everything 2002 : Movies ( Schickel ) '' . Time . Time Inc . Retrieved February 17 , 2014 .
</item 1>

<item 2>
8 Mile (film)
He praised Hanson 's directing and the performances , and compared the final battle with Papa Doc to the fight between Rocky Balboa and Apollo Creed in Rocky .
Top lists ( edit ) 8 Mile has been named in various year - end and all - time top lists : 2nd -- Billboard ( Erika Ramirez ) : Top 10 Best Hip - Hop Movies Ever
7th -- The New York Observer ( Andrew Sarris ) : The 10 Best English - Language Films of 2002 9th -- Time ( Richard Schickel ) : Top 10 Movies of 2002 10th --
Rolling Stone ( Peter Travers ) : The Best Movies of 2002 N / A -- The Daily Californian : Best Films of 2002 Awards and honors ( edit ) In 2003 , Eminem won
the Academy Award for Best Original Song at the 75th Academy Awards , for his single `` Lose Yourself '' from the soundtrack of 8 Mile . Thus he became the
first hip hop artist ever to win an Academy Award . He was not present at the ceremony , but musician Luis Resto accepted the award .
</item 2>

<item 3>
8 Mile (film)
In all rounds of the rap battle , Jimmy has to compete against one member of `` the Leaders . '' After winning the first two rounds , he has to challenge Papa
Doc . He wins by exposing his opponent 's weaknesses as well as his own . With nothing to say in rebuttal , Papa Doc hands the microphone back to Future ,
embarrassed . After being congratulated by Alex and his friends , Jimmy is offered a position by Future , hosting battles at The Shelter . Jimmy declines ,
saying he has to get back to work and to find success his own way . Cast ( edit ) Eminem as Jimmy `` B - Rabbit '' Smith , Jr .
</item 3>

<item 4>
8 Mile (film)
The site 's consensus is that `` Even though the story is overly familiar , there 's enough here for an engaging ride . '' On Metacritic , which assigns a
weighted mean rating out of 100 reviews from film critics , the film has rating score of 77 based on 38 reviews , which indicates `` generally favorable reviews
'' . CinemaScore polls conducted during the opening weekend revealed the average grade cinemagoers gave 8 Mile was B+ on an A+ to F scale , with the core under
- 21 demographics giving it an A . Roger Ebert gave the film 3 out of 4 stars . He said that we `` are hardly started in 8 Mile , and already we see that this
movie stands aside from routine debut films by pop stars '' and that it is `` a faithful reflection of his myth '' .
</item 4>

<item 5>
8 Mile (film)
Jimmy works at a car factory . When he asks for extra shifts , his supervisor laughs and dismisses his request on account of his habitual lateness . Jimmy
befriends a woman named Alex . Over time , Jimmy begins to take more responsibility for the direction of his life . When he exhibits an improved attitude and
performance at work , his supervisor grants him the extra shifts he requested . Stephanie receives a notice evicting her and the family as she can not pay rent
. Despite Stephanie 's best attempts to keep the eviction notice a secret , Greg finds out about it . A fight ensues between him and Jimmy , and Greg leaves
Stephanie for good . Jimmy 's friendship with Wink , a radio DJ with ties to a record label promoter , becomes strained after he discovers that Wink does
promotional work for Jimmy 's rivals , a rap group known as the `` Leaders of the Free World . '' At one point , Jimmy and his friends get into a violent brawl
with the Leaders , which is disrupted when Jimmy 's friend Cheddar Bob pulls out a gun and accidentally shoots himself in the leg ; he survives but is confined
to crutches .
</item 5>

<item 6>
8 Mile (film)
8 Mile ( film ) - wikipedia 8 Mile ( film ) Jump to : navigation , search 8 Mile Theatrical release poster Directed by Curtis Hanson Produced by Curtis Hanson
Brian Grazer Jimmy Iovine Written by Scott Silver Starring Eminem Kim Basinger Brittany Murphy Mekhi Phifer Music by Eminem Cinematography Rodrigo Prieto Edited
by Jay Rabinowitz Production company Imagine Entertainment Distributed by Universal Pictures Release date November 8 , 2002 ( 2002 - 11 - 08 ) Running time 110
minutes Country United States Language English Budget $41 million Box office $242.9 million 8 Mile is a 2002 American hip hop drama film written by Scott Silver
, directed by Curtis Hanson , and starring Eminem , Mekhi Phifer , Brittany Murphy , Michael Shannon , and Kim Basinger . The film is based loosely on Eminem 's
actual upbringing , and follows white rapper B - Rabbit ( Eminem ) and his attempt to launch a career in a genre dominated by African - Americans .
</item 6>

<item 7>
8 Mile (film)
Mekhi Phifer as Future , Jimmy 's closest friend and the rap battle host Brittany Murphy as Alex Latourno , Jimmy 's love interest Kim Basinger as Stephanie ,
Jimmy and Lily 's mother Anthony Mackie as Papa Doc Michael Shannon as Greg Buehl , Stephanie 's boyfriend Eugene Byrd as Wink Evan Jones as Cheddar Bob , Jimmy
's friend Omar Benson Miller as Sol George , Jimmy 's friend and Iz 's brother De'Angelo Wilson as DJ Iz , Jimmy 's friend and Sol 's brother Taryn Manning as
Janeane , Jimmy 's ex-girlfriend Proof as Lil ' Tic Xzibit as Mike ( Male Lunch Truck Rapper ) Craig Chandler as Paul Obie Trice as Male parking lot rapper
Chloe Greenfield as Lily , Jimmy 's sister John Singleton as Bouncer Miz Korona as Vanessa ( Female Lunch Truck Rapper ) Brandon T. Jackson ( uncredited ) as a
Chin Tiki club - foe Music ( edit ) Main article : Music from and Inspired by the Motion Picture 8 Mile Music from and Inspired by the Motion Picture 8 Mile is
the soundtrack to 8 Mile .
</item 7>

<item 8>
8 Mile (film)
He said that Eminem , as an actor , is `` convincing without being too electric '' and `` survives the X-ray truth - telling of the movie camera '' In the At
the Movies with Ebert and Roeper review , both Ebert and Richard Roeper gave the film a thumbs up ; Roeper said that Eminem has a `` winning screen presence ''
and `` raw magic '' to him . He was happy with Rabbit 's `` tender side '' presented through his relationship with the `` adorable '' Greenfield as his sister ,
but felt that Basinger was `` really miscast '' . Roeper said : `` 8 Mile probably wo n't win converts to rap , but it should thrill Eminem fans . '' Peter
Travers gave the film 3.5 out of 4 stars . He said that 8 Mile `` is a real movie , not a fast - buck package to exploit the fan base of a rap nonentity '' that
`` qualifies as a cinematic event by tapping into the roots of Eminem and the fury and feeling that inform his rap . ''
</item 8>

<item 9>
8 Mile (film)
Ronnie Specter , Matiki Anoff MTV Movie Awards Best Movie Nominated 8 Mile Best Male Performance Won Eminem Breakthrough Male Performance Won Eminem Online Film
Critics Society Awards Best Breakthrough Performance Nominated Eminem Phoenix Film Critics Society Awards Best Original Song -- Lose Yourself Nominated Eminem
Satellite Awards Best Original Song -- Lose Yourself Nominated Eminem Teen Choice Awards Choice Movie : Drama Nominated 8 Mile Choice Movie Actor : Drama Won
Eminem Choice Movie : Male Breakout Star Won Eminem Choice Crossover Artist Nominated Eminem Choice Movie : Liplock Nominated Eminem and Brittany Murphy World
Soundtrack Awards Best Original Song Written for a Film -- Lose Yourself Nominated Eminem ASCAP Awards Most Performed Song from a Motion Picture -- Lose
Yourself Won Eminem Grammy Award Grammy Award for Record of the Year -- Lose Yourself Nominated Eminem Grammy Award for Song of the Year -- Lose Yourself
Nominated Jeff Bass , Eminem & Luis Resto Grammy Award for Best Rap Song -- Lose Yourself Won Jeff Bass , Eminem & Luis Resto Grammy Award for Best Male Rap
Solo Performance -- Lose Yourself Won Eminem Grammy Award for Best Song Written for Visual Media -- Lose Yourself Nominated Jeff Bass ,
</item 9>

<item 10>
8 Mile (film)
The title is derived from 8 Mile Road , the highway between the predominantly black city of Detroit and Wayne County and the predominantly White Oakland County
and Macomb County suburbs . It was filmed mostly on location . 8 Mile was a critical and commercial success . Eminem won the Academy Award for Best Original
Song for `` Lose Yourself . '' Contents ( hide ) 1 Plot 2 Cast 3 Music 4 Reception 4.1 Box office 4.2 Critical reception 4.3 Top lists 4.4 Awards and honors 5
References 6 External links Plot ( edit ) In 1995 in Detroit , Jimmy `` B - Rabbit '' Smith Jr. is a blue - collar worker from a poor family . He has moved back
north of 8 Mile Road to the run - down trailer home in Warren , Michigan of his alcoholic mother Stephanie , his little sister Lily , and Stephanie 's abusive
live - in boyfriend Greg . Although encouraged by his friends , Jimmy worries about his potential as a rapper . He falters during a rap battle one night at a
local venue , the Shelter , and he leaves the stage humiliated . Jimmy works at a car factory .
</item 10>
=== END ITEM id=8c06e1c343d30d6b ===

=== ITEM id=a057620d4a3b7ef1 ===
Gold sentences and facts:
Sentence 0: State Pre-Prohibition ( prior to 1919 ) Post-Prohibition ( after 1933 ) 1970s / 26th Amendment ( adopted in 1971 ) 1980s / Drinking Age Act of 1984 21st century Alabama Pre 1881 : None 1881 : 21 ( none with consent of parent ) 21 1975 : Lowered to 19 1986 : Raised to 21 21 ( no one underage is allowed consumption Section 28 - 1 ) Alaska ? 21 1970 : Lowered to 19 1984 : Raised to 21 with grandfather clause ( if born before January 1 , 1965 ) 21 Arizona ? 21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and liquor 18 Raised to 21 in 1987 21 Connecticut ? 21 1972 : Lowered to 18 1982 : Raised to 19 1983 : Raised to 20 1985 : Raised to 21 21 ( exemptions : ( 1 ) a person over age eighteen who is an employee or permit holder under section 30 - 90a and who possesses alcoholic liquor in the course of such person 's employment or business , ( 2 ) a minor who possesses alcoholic liquor on the order of a practicing physician , or ( 3 ) a minor who possesses alcoholic liquor while accompanied by a parent , guardian or spouse of the minor , who has attained the age of twenty - one . Nothing in this subsection shall be construed to burden a person 's exercise of religion under section 3 of article first of the Constitution of the state in violation of subsection ( a ) of section 52 - 571b . Delaware ? 21 1972 : Lowered to 20 1984 : Raised to 21 21 District of Columbia ? 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 : Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 : Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to 19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ? Post-1934 : 21 21 21 21 Iowa ? 21 1972 : Lowered to 19 in 1972 ( Jul 1 ) 1973 : Lowered to 18 in 1973 ( Jul 1 ) 1978 : Raised to 19 in 1978 ( Jul 1 ) 1986 : Raised to 21 in 1986 ( Jul 1 ) with grandfather clause . 21 Kansas ? 1949 : 18 : for beer and wine with less than 4 % ABV 21 : for liquor 18 : for beer and wine with less than 4 % ABV 21 : for beer and wine with 4 % ABV or more 21 : for liquor 1985 : Raised to 21 21 ( However , anyone under 21 may consume cereal malt beverages with parental supervision on their own property ) Kentucky ? 1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 : De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision , raising to 21 three months later . Other exceptions still remain . 21 ( Exceptions to state law include : For an established religious purpose ; When a person under twenty - one years of age is accompanied by a parent , spouse , or legal guardian twenty - one years of age or older ; For medical purposes when purchased as an over the counter medication , or when prescribed or administered by a licensed physician , pharmacist , dentist , nurse , hospital , or medical institution ; In a private residence , which shall include a residential dwelling and up to twenty contiguous acres , on which the dwelling is located , owned by the same person who owns the dwelling ; The sale , handling , transport , or service in dispensing of any alcoholic beverage pursuant to lawful ownership of an establishment or to lawful employment of a person under twenty - one years of age by a duly licensed manufacturer , wholesaler , or retailer of beverage alcohol . ) Maine ? 21 1969 : Lowered to 20 1972 : Lowered to 18 1977 : Raised to 20 1985 : Raised to 21 21 Maryland ? 21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses , teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18 in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21 Mississippi ? 1966 : 18 : for beer and wine 21 : for liquor ( alcohol not legalized until 1966 ) 18 Raised to 21 in 1986 21 Missouri ? 1945 : 21 21 21 21 Montana ? 21 1971 : Lowered to 19 1973 : Lowered to 18 1979 : Raised to 19 1987 : Raised to 21 21 Nebraska ? 21 1969 : Lowered to 20 1972 : Lowered to 19 1980 : Raised to 20 1985 : Raised to 21 21 Nevada ? 1933 : 21 21 21 21 New Hampshire ? 21 1973 : Lowered to 18 1979 : Raised to 20 1985 : Raised to 21 21 New Jersey Before 1880 : None Post-1880 : 18 ( penalties only against businesses ) 21 1973 : Lowered to 18 1980 : Raised to 19 1983 : Raised to 21 21 New Mexico ? 1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 : for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ? 1935 : 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : beer 21 : wine and liquor 1988 : 21 21 ( However , anyone can drink with parents or legal - age spouse ) Oklahoma ? Men : 21 Women : 18 : 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1976 : Lowered to 18 ( for 3.2 % ABW beer ) for both sexes in 1976 ( Dec ) Last state to lower the drinking age . 1983 : Raised to 21 21 Oregon ? 1933 : 21 21 21 21 Pennsylvania ? 1935 : 21 21 21 21 Rhode Island ? 21 1970 : Lowered to 18 1980 : Raised to 19 1981 : Raised to 20 1984 : Raised to 21 21 South Carolina 21 1935 : 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 21 South Dakota ? 1934 : 21 ( all ) 1939 : Lowered to 18 : for 3.2 % beer Raised to 19 : for 3.2 % beer 1972 : 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1988 : Raised to 21 21 Tennessee Pre 1881 : None 1881 : 21 21 1964 : Lowered to 18 1979 : Raised to 19 1984 : Raised to 21 21 Texas Pre 1909 : 21 None ( with written permission of parent or guardian ) Post 1909 : 21 21 1973 : Lowered to 18 1981 : Raised to 19 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 Utah ? 1935 : 21 21 21 21 Vermont ? 21 1971 : Lowered to 18 1986 : Raised to 21 21 Virginia ? 21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer ) 1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all ) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all ) 18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales , remained 18 for on - premises ) . 1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 ( however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010
  Fact 0.0: The table lists each state's drinking age in five periods: pre-Prohibition (prior to 1919), post-Prohibition (after 1933), the 1970s (26th Amendment, adopted in 1971), the 1980s (Drinking Age Act of 1984), and the 21st century.
  Fact 0.1: Alabama had no drinking age before 1881.
  Fact 0.2: Alabama set its drinking age at 21 in 1881, with none with consent of parent.
  Fact 0.3: Alabama's drinking age was 21 after Prohibition.
  Fact 0.4: Alabama lowered its drinking age to 19 in 1975.
  Fact 0.5: Alabama raised its drinking age to 21 in 1986.
  Fact 0.6: Alabama's 21st-century drinking age is 21, and no one underage is allowed consumption under Section 28-1.
  Fact 0.7: Alaska's post-Prohibition drinking age was 21.
  Fact 0.8: Alaska lowered its drinking age to 19 in 1970.
  Fact 0.9: Alaska raised its drinking age to 21 in 1984, with a grandfather clause for those born before January 1, 1965.
  Fact 0.10: Alaska's 21st-century drinking age is 21.
  Fact 0.11: Arizona's post-Prohibition drinking age was 21.
  Fact 0.12: Arizona lowered its drinking age to 19 in 1972.
  Fact 0.13: Arizona raised its drinking age to 21 in 1985.
  Fact 0.14: Arizona's 21st-century drinking age is 21.
  Fact 0.15: Arkansas had no drinking age before 1925.
  Fact 0.16: Arkansas set its drinking age at 21 in 1925.
  Fact 0.17: Arkansas's drinking age stayed at 21 in the post-Prohibition, 1970s, 1980s and 21st-century periods.
  Fact 0.18: California was regulated by municipality or county before 1891, when the common age was 16.
  Fact 0.19: California set a statewide drinking age of 18 in 1891.
  Fact 0.20: California set its drinking age at 21 in 1933.
  Fact 0.21: California's drinking age stayed at 21 after 1933, except small amounts for religious ceremonies.
  Fact 0.22: Colorado had no drinking age before Prohibition.
  Fact 0.23: Colorado set its drinking age in 1945 at 18 for beer and 21 for wine and liquor.
  Fact 0.24: Colorado's drinking age was 18 in the 1970s.
  Fact 0.25: Colorado raised its drinking age to 21 in 1987.
  Fact 0.26: Colorado's 21st-century drinking age is 21.
  Fact 0.27: Connecticut's post-Prohibition drinking age was 21.
  Fact 0.28: Connecticut lowered its drinking age to 18 in 1972.
  Fact 0.29: Connecticut raised its drinking age to 19 in 1982.
  Fact 0.30: Connecticut raised its drinking age to 20 in 1983.
  Fact 0.31: Connecticut raised its drinking age to 21 in 1985.
  Fact 0.32: Connecticut's 21st-century drinking age is 21.
  Fact 0.33: Connecticut exempts a person over 18 who is an employee or permit holder under section 30-90a and who possesses alcoholic liquor in the course of such employment or business.
  Fact 0.34: Connecticut exempts a minor who possesses alcoholic liquor on the order of a practicing physician.
  Fact 0.35: Connecticut exempts a minor who possesses alcoholic liquor while accompanied by a parent, guardian or spouse of the minor who has attained the age of 21.
  Fact 0.36: Connecticut's age limit does not burden a person's exercise of religion under section 3 of article first of the state Constitution in violation of section 52-571b(a).
  Fact 0.37: Delaware's post-Prohibition drinking age was 21.
  Fact 0.38: Delaware lowered its drinking age to 20 in 1972.
  Fact 0.39: Delaware raised its drinking age to 21 in 1984.
  Fact 0.40: Delaware's 21st-century drinking age is 21.
  Fact 0.41: The District of Columbia set its drinking age at 18 for beer and wine and 21 for liquor.
  Fact 0.42: The District of Columbia's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.43: The District of Columbia raised its drinking age to 21 in 1986, with a grandfather clause.
  Fact 0.44: The District of Columbia's 21st-century drinking age is 21.
  Fact 0.45: Florida's drinking age was 18 in the 1970s.
  Fact 0.46: Florida raised its drinking age to 19 in 1980.
  Fact 0.47: Florida raised its drinking age to 21 in 1985, with a grandfather clause.
  Fact 0.48: Florida's 21st-century drinking age is 21.
  Fact 0.49: Georgia's post-Prohibition drinking age was 21.
  Fact 0.50: Georgia lowered its drinking age to 18 in 1972.
  Fact 0.51: Georgia raised its drinking age to 19 in 1984.
  Fact 0.52: Georgia raised its drinking age to 20 in 1985.
  Fact 0.53: Georgia raised its drinking age to 21 in 1986.
  Fact 0.54: Georgia's 21st-century drinking age is 21.
  Fact 0.55: Hawaii's post-Prohibition drinking age was 21.
  Fact 0.56: Hawaii lowered its drinking age to 18 in 1972.
  Fact 0.57: Hawaii raised its drinking age to 21 in 1986.
  Fact 0.58: Hawaii's 21st-century drinking age is 21.
  Fact 0.59: Idaho's post-Prohibition drinking age was 20 for beer and 21 for wine and liquor.
  Fact 0.60: Idaho lowered its drinking age to 19 for all alcohol in 1972.
  Fact 0.61: Idaho raised its drinking age to 21 on April 11, 1987, with a grandfather clause.
  Fact 0.62: Idaho's 21st-century drinking age is 21.
  Fact 0.63: Illinois had a drinking age equal to the age of majority in 1872, with none with written consent of parent.
  Fact 0.64: Illinois's drinking age was the age of majority before 1961.
  Fact 0.65: Illinois raised its drinking age to 21 in 1961.
  Fact 0.66: Illinois lowered its drinking age to 19 for beer and wine in 1973, and kept 21 for liquor.
  Fact 0.67: Illinois raised its drinking age to 21 in 1980.
  Fact 0.68: Illinois's 21st-century drinking age is 21.
  Fact 0.69: Indiana's drinking age has been 21 since 1934.
  Fact 0.70: Indiana's drinking age was 21 in the 1970s, 1980s and 21st century.
  Fact 0.71: Iowa's post-Prohibition drinking age was 21.
  Fact 0.72: Iowa lowered its drinking age to 19 on July 1, 1972.
  Fact 0.73: Iowa lowered its drinking age to 18 on July 1, 1973.
  Fact 0.74: Iowa raised its drinking age to 19 on July 1, 1978.
  Fact 0.75: Iowa raised its drinking age to 21 on July 1, 1986, with a grandfather clause.
  Fact 0.76: Iowa's 21st-century drinking age is 21.
  Fact 0.77: Kansas set its post-Prohibition drinking age in 1949 at 18 for beer and wine under 4% ABV and 21 for liquor.
  Fact 0.78: Kansas's 1970s drinking age was 18 for beer and wine under 4% ABV, and 21 for beer and wine at 4% ABV or more and for liquor.
  Fact 0.79: Kansas raised its drinking age to 21 in 1985.
  Fact 0.80: Kansas's 21st-century drinking age is 21, but anyone under 21 may consume cereal malt beverages with parental supervision on their own property.
  Fact 0.81: Kentucky's drinking age was 21 from 1938 onward.
  Fact 0.82: Louisiana's drinking age was 18 as of 1948.
  Fact 0.83: Louisiana's drinking age was 18 in the 1970s.
  Fact 0.84: Louisiana's drinking age was raised to 21 de jure in 1987, but the de facto age stayed at 18 until 1995 because of a sale loophole.
  Fact 0.85: Louisiana's de facto drinking age was raised to 21 in 1995 when the loophole was closed.
  Fact 0.86: In 1996 the Louisiana Supreme Court briefly lowered the drinking age to 18, until it reversed its decision and raised it to 21 three months later.
  Fact 0.87: Louisiana's 21st-century drinking age is 21, with exceptions that remain.
  Fact 0.88: Louisiana's drinking age exception covers an established religious purpose.
  Fact 0.89: Louisiana's drinking age exception covers a person under 21 accompanied by a parent, spouse, or legal guardian who is 21 or older.
  Fact 0.90: Louisiana's drinking age exception covers medical purposes, when purchased as an over-the-counter medication or when prescribed or administered by a licensed physician, pharmacist, dentist, nurse, hospital, or medical institution.
  Fact 0.91: Louisiana's drinking age exception covers a private residence, including a residential dwelling and up to twenty contiguous acres on which the dwelling is located, owned by the same person who owns the dwelling.
  Fact 0.92: Louisiana's drinking age exception covers the sale, handling, transport, or service in dispensing of alcoholic beverages under lawful ownership of an establishment or lawful employment of a person under 21 by a duly licensed manufacturer, wholesaler, or retailer of beverage alcohol.
  Fact 0.93: Maine's post-Prohibition drinking age was 21.
  Fact 0.94: Maine lowered its drinking age to 20 in 1969.
  Fact 0.95: Maine lowered its drinking age to 18 in 1972.
  Fact 0.96: Maine raised its drinking age to 20 in 1977.
  Fact 0.97: Maine raised its drinking age to 21 in 1985.
  Fact 0.98: Maine's 21st-century drinking age is 21.
  Fact 0.99: Maryland's post-Prohibition drinking age was 21.
  Fact 0.100: Maryland lowered its drinking age to 18 for beer and wine in 1974, and kept 21 for liquor.
  Fact 0.101: Maryland raised its drinking age to 21 in 1982, with a grandfather clause.
  Fact 0.102: Maryland's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.103: Massachusetts's post-Prohibition drinking age was 21.
  Fact 0.104: Massachusetts lowered its drinking age to 18 in 1973.
  Fact 0.105: Massachusetts raised its drinking age to 20 on April 16, 1979.
  Fact 0.106: Massachusetts raised its drinking age to 21 in 1985.
  Fact 0.107: Massachusetts's 21st-century drinking age is 21.
  Fact 0.108: Michigan's post-Prohibition drinking age was 21.
  Fact 0.109: Michigan lowered its drinking age to 18 on January 1, 1972.
  Fact 0.110: Michigan raised its drinking age to 19 on December 3, 1978.
  Fact 0.111: Michigan raised its drinking age to 21 on December 21, 1978, 18 days later.
  Fact 0.112: Michigan was the first state to raise its age to 21 after lowering it.
  Fact 0.113: Michigan's 21st-century drinking age is 21.
  Fact 0.114: Minnesota's post-Prohibition drinking age was 21.
  Fact 0.115: Minnesota lowered its drinking age to 18 on June 1, 1973.
  Fact 0.116: Minnesota raised its drinking age to 19 on September 1, 1976.
  Fact 0.117: Minnesota was the first state to raise its drinking age after lowering it.
  Fact 0.118: Minnesota raised its drinking age to 21 in 1986.
  Fact 0.119: Minnesota's 21st-century drinking age is 21.
  Fact 0.120: Mississippi set a drinking age of 18 for beer and wine and 21 for liquor in 1966, when alcohol was legalized.
  Fact 0.121: Mississippi's drinking age was 18 in the 1970s.
  Fact 0.122: Mississippi raised its drinking age to 21 in 1986.
  Fact 0.123: Mississippi's 21st-century drinking age is 21.
  Fact 0.124: Missouri's drinking age was 21 from 1945 onward.
  Fact 0.125: Montana's post-Prohibition drinking age was 21.
  Fact 0.126: Montana lowered its drinking age to 19 in 1971.
  Fact 0.127: Montana lowered its drinking age to 18 in 1973.
  Fact 0.128: Montana raised its drinking age to 19 in 1979.
  Fact 0.129: Montana raised its drinking age to 21 in 1987.
  Fact 0.130: Montana's 21st-century drinking age is 21.
  Fact 0.131: Nebraska's post-Prohibition drinking age was 21.
  Fact 0.132: Nebraska lowered its drinking age to 20 in 1969.
  Fact 0.133: Nebraska lowered its drinking age to 19 in 1972.
  Fact 0.134: Nebraska raised its drinking age to 20 in 1980.
  Fact 0.135: Nebraska raised its drinking age to 21 in 1985.
  Fact 0.136: Nebraska's 21st-century drinking age is 21.
  Fact 0.137: Nevada's drinking age was 21 from 1933 onward.
  Fact 0.138: New Hampshire's post-Prohibition drinking age was 21.
  Fact 0.139: New Hampshire lowered its drinking age to 18 in 1973.
  Fact 0.140: New Hampshire raised its drinking age to 20 in 1979.
  Fact 0.141: New Hampshire raised its drinking age to 21 in 1985.
  Fact 0.142: New Hampshire's 21st-century drinking age is 21.
  Fact 0.143: New Jersey had no drinking age before 1880.
  Fact 0.144: New Jersey set a drinking age of 18 after 1880, with penalties only against businesses.
  Fact 0.145: New Jersey's post-Prohibition drinking age was 21.
  Fact 0.146: New Jersey lowered its drinking age to 18 in 1973.
  Fact 0.147: New Jersey raised its drinking age to 19 in 1980.
  Fact 0.148: New Jersey raised its drinking age to 21 in 1983.
  Fact 0.149: New Jersey's 21st-century drinking age is 21.
  Fact 0.150: New Mexico's drinking age was 21 from 1934 onward.
  Fact 0.151: New York's pre-Prohibition drinking age was 21.
  Fact 0.152: New York's post-Prohibition drinking age was 18.
  Fact 0.153: New York's drinking age was 18 in the 1970s.
  Fact 0.154: New York raised its drinking age to 19 in December 1982.
  Fact 0.155: New York raised its drinking age to 21 in December 1985.
  Fact 0.156: New York's 21st-century drinking age is 21.
  Fact 0.157: North Carolina's post-Prohibition drinking age was 18.
  Fact 0.158: North Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.159: North Carolina's 1983 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.160: North Carolina raised its drinking age to 21 on September 1, 1986.
  Fact 0.161: North Carolina's 21st-century drinking age is 21.
  Fact 0.162: North Dakota's drinking age was 21 from 1936 onward.
  Fact 0.163: Ohio's pre-Prohibition drinking age is unknown.
  Fact 0.164: Ohio set its post-Prohibition drinking age in 1935 at 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.165: Ohio's 1970s drinking age was 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.166: In 1982 Ohio set the age at 19 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.167: A second 1982 Ohio entry set the age at 19 for beer and 21 for wine and liquor.
  Fact 0.168: Ohio raised its drinking age to 21 in 1988.
  Fact 0.169: Ohio's 21st-century drinking age is 21, and anyone can drink with parents or a legal-age spouse.
  Fact 0.170: Oklahoma's drinking age was 21 for men and 18 for women for 3.2% ABW beer, and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.171: Oklahoma lowered its drinking age to 18 for 3.2% ABW beer for both sexes in December 1976.
  Fact 0.172: Oklahoma was the last state to lower the drinking age.
  Fact 0.173: Oklahoma raised its drinking age to 21 in 1983.
  Fact 0.174: Oklahoma's 21st-century drinking age is 21.
  Fact 0.175: Oregon's drinking age was 21 from 1933 onward.
  Fact 0.176: Pennsylvania's drinking age was 21 from 1935 onward.
  Fact 0.177: Rhode Island's post-Prohibition drinking age was 21.
  Fact 0.178: Rhode Island lowered its drinking age to 18 in 1970.
  Fact 0.179: Rhode Island raised its drinking age to 19 in 1980.
  Fact 0.180: Rhode Island raised its drinking age to 20 in 1981.
  Fact 0.181: Rhode Island raised its drinking age to 21 in 1984.
  Fact 0.182: Rhode Island's 21st-century drinking age is 21.
  Fact 0.183: South Carolina's pre-Prohibition drinking age was 21.
  Fact 0.184: South Carolina set its post-Prohibition drinking age in 1935 at 18 for beer and wine and 21 for liquor.
  Fact 0.185: South Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.186: South Carolina's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.187: South Carolina raised its drinking age to 21 in 1986.
  Fact 0.188: South Carolina's 21st-century drinking age is 21.
  Fact 0.189: South Dakota's drinking age was 21 for all alcohol in 1934.
  Fact 0.190: South Dakota lowered its drinking age to 18 for 3.2% beer in 1939 and raised it to 19 for 3.2% beer.
  Fact 0.191: South Dakota's 1972 drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.192: South Dakota's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.193: South Dakota raised its drinking age to 21 in 1988.
  Fact 0.194: South Dakota's 21st-century drinking age is 21.
  Fact 0.195: Tennessee had no drinking age before 1881.
  Fact 0.196: Tennessee set its drinking age at 21 in 1881.
  Fact 0.197: Tennessee's post-Prohibition drinking age was 21.
  Fact 0.198: Tennessee lowered its drinking age to 18 in 1964.
  Fact 0.199: Tennessee raised its drinking age to 19 in 1979.
  Fact 0.200: Tennessee raised its drinking age to 21 in 1984.
  Fact 0.201: Tennessee's 21st-century drinking age is 21.
  Fact 0.202: Texas's drinking age was 21 before 1909, with none with written permission of parent or guardian.
  Fact 0.203: Texas's drinking age was 21 after 1909 and after Prohibition.
  Fact 0.204: Texas lowered its drinking age to 18 in 1973.
  Fact 0.205: Texas raised its drinking age to 19 in 1981.
  Fact 0.206: Texas raised its drinking age to 21 on September 1, 1986.
  Fact 0.207: Texas's 21st-century drinking age is 21.
  Fact 0.208: Utah's drinking age was 21 from 1935 onward.
  Fact 0.209: Vermont's post-Prohibition drinking age was 21.
  Fact 0.210: Vermont lowered its drinking age to 18 in 1971.
  Fact 0.211: Vermont raised its drinking age to 21 in 1986.
  Fact 0.212: Vermont's 21st-century drinking age is 21.
  Fact 0.213: Virginia's post-Prohibition drinking age was 21.
  Fact 0.214: Virginia lowered its drinking age to 18 for beer and wine in 1974, while 21 for liquor remained.
  Fact 0.215: Virginia raised its drinking age to 19 for off-premises consumption in 1981.
  Fact 0.216: Virginia raised its drinking age to 19 for all beer in 1983.
  Fact 0.217: Virginia raised its drinking age to 21 in 1985.
  Fact 0.218: Virginia's 21st-century drinking age is 21.
  Fact 0.219: Washington's drinking age has been the age of majority since 1877.
  Fact 0.220: Washington's drinking age was 21 from after Prohibition onward.
  Fact 0.221: West Virginia's post-Prohibition drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.222: West Virginia lowered its drinking age to 18 for all alcohol in 1972.
  Fact 0.223: West Virginia raised its drinking age to 19 in 1983, and to 21 for non-residents.
  Fact 0.224: West Virginia raised its drinking age to 21 in 1986.
  Fact 0.225: West Virginia's 21st-century drinking age is 21.
  Fact 0.226: Wisconsin's drinking age was 18 for wine and liquor, with none for beer, from 1839 to 1866.
  Fact 0.227: Wisconsin set its drinking age at 21 for all alcohol after 1866.
  Fact 0.228: Wisconsin's post-Prohibition drinking age was 18 for beer and 21 for wine and spirits.
  Fact 0.229: In 1957 Wisconsin set a 21 limit for residents of bordering states with an age limit of 21.
  Fact 0.230: In 1963 Wisconsin raised its age to 21 for off-premises beer sales, while it remained 18 for on-premises.
  Fact 0.231: Wisconsin lowered its drinking age to 18 for all alcohol in March 1972.
  Fact 0.232: Wisconsin lifted the border state restriction in 1977.
  Fact 0.233: Wisconsin raised its drinking age to 19 in 1984.
  Fact 0.234: Wisconsin raised its drinking age to 21 on September 1, 1986.
  Fact 0.235: Wisconsin's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.236: Wyoming's post-Prohibition drinking age was 21.
  Fact 0.237: Wyoming lowered its drinking age to 19 in 1973.
  Fact 0.238: Wyoming raised its drinking age to 21 on July 1, 1988.
  Fact 0.239: Wyoming was the last state to raise its de jure drinking age to 21.
  Fact 0.240: Wyoming's 21st-century drinking age is 21.
  Fact 0.241: American Samoa's listed drinking age is 21.
  Fact 0.242: The Northern Mariana Islands' listed drinking age is 21.
  Fact 0.243: Puerto Rico's post-Prohibition drinking age was 18.
  Fact 0.244: Puerto Rico's 1970s drinking age was 18.
  Fact 0.245: Puerto Rico kept its drinking age at 18 despite a 10% highway funding penalty under the Drinking Age Act.
  Fact 0.246: Puerto Rico's 21st-century drinking age is 18.
  Fact 0.247: The Virgin Islands' drinking age was 18 in the 1980s and in the 21st century.
  Fact 0.248: Guam's listed drinking age includes 21 and 18.
  Fact 0.249: Guam's 1980s and 21st-century drinking age was 18.
  Fact 0.250: Guam raised its drinking age to 21 in 2010.
Sentence 1: From 1976 to 1983 , several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 % of their federal highway funds . By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
  Fact 1.0: From 1976 to 1983, several states voluntarily raised their purchase ages to 19 (or, less commonly, 20 or 21).
  Fact 1.1: The voluntary purchase-age increases from 1976 to 1983 were in part to combat drunk driving fatalities.
  Fact 1.2: In 1984 Congress passed the National Minimum Drinking Age Act.
  Fact 1.3: The National Minimum Drinking Age Act required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10% of their federal highway funds.
  Fact 1.4: By mid-1988, all 50 states and the District of Columbia had raised their purchase ages to 21.
  Fact 1.5: Puerto Rico, Guam, and the Virgin Islands had not raised their purchase ages to 21 by mid-1988.
  Fact 1.6: South Dakota and Wyoming were the final two states to comply with the age 21 mandate.
  Fact 1.7: The current drinking age of 21 remains a point of contention among many Americans.
  Fact 1.8: The current drinking age of 21 is higher than the age of majority, which is 18 in most states.
  Fact 1.9: The current drinking age of 21 is higher than the drinking ages of most other countries.
  Fact 1.10: The National Minimum Drinking Age Act is seen as a congressional sidestep of the tenth amendment.
  Fact 1.11: Debates over the drinking age have not been highly publicized.
  Fact 1.12: A few states have proposed legislation to lower their drinking age.
  Fact 1.13: Guam raised its drinking age to 21 in July 2010.

Retrieved items:
<item 1>
U.S. history of alcohol minimum purchase age by state
1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 :
for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ?
</item 1>

<item 2>
U.S. history of alcohol minimum purchase age by state
By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see
Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point
of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other
countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly
publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
</item 2>

<item 3>
U.S. history of alcohol minimum purchase age by state
1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 (
however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to
raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway
funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010 Additional Notes ( edit ) Contrary to popular
belief , since the National Minimum Drinking Age Act of 1984 , not all states specifically prohibit minors ' and young adults ' consumption of alcohol in
private settings .
</item 3>

<item 4>
U.S. history of alcohol minimum purchase age by state
1934 : 21 ( all ) 1939 : Lowered to 18 : for 3.2 % beer Raised to 19 : for 3.2 % beer 1972 : 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and
wine 21 : for liquor 1988 : Raised to 21 21 Tennessee Pre 1881 : None 1881 : 21 21 1964 : Lowered to 18 1979 : Raised to 19 1984 : Raised to 21 21 Texas Pre
1909 : 21 None ( with written permission of parent or guardian ) Post 1909 : 21 21 1973 : Lowered to 18 1981 : Raised to 19 1986 : Raised to 21 in 1986 ( Sep 1
) 21 Utah ? 1935 : 21 21 21 21 Vermont ? 21 1971 : Lowered to 18 1986 : Raised to 21 21 Virginia ?
</item 4>

<item 5>
U.S. history of alcohol minimum purchase age by state
18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 :
Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 :
Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with
grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to
19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ?
</item 5>

<item 6>
U.S. history of alcohol minimum purchase age by state
Shortly after the ratification of the 21st amendment in December , most states set their purchase ages at 21 since that was the Voting age at the time . Most of
these limits remained constant until the early 1970s . From 1969 to 1976 , some 30 states lowered their purchase ages , generally to 18 . This was primarily
because the voting age was lowered from 21 to 18 in 1971 with the 26th amendment . A lot of states started to lower their minimum drinking age in response ,
most of this occurring in 1972 or 1973 . Twelve states kept their purchase ages at 21 since repeal of Prohibition and never changed them . From 1976 to 1983 ,
several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress
passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 %
of their federal highway funds .
</item 6>

<item 7>
U.S. history of alcohol minimum purchase age by state
21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common
age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and
liquor 18 Raised to 21 in 1987 21 Connecticut ?
</item 7>

<item 8>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer )
1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all
) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all )
18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales ,
remained 18 for on - premises ) .
</item 8>

<item 9>
After Prohibition’s repeal in 1933, most states set alcohol purchase ages at 21. Following the voting-age change in 1971, about 30 states lowered them—often to
18—during 1969–76. Many later raised ages amid drunk-driving concerns. The 1984 federal law tied highway funding to a purchase and public-possession age of 21;
all states and D.C. complied by 1988. Some states had different ages by beverage, and
</item 9>

<item 10>
U.S. history of alcohol minimum purchase age by state
1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 :
De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision ,
raising to 21 three months later . Other exceptions still remain .
</item 10>
=== END ITEM id=a057620d4a3b7ef1 ===

=== ITEM id=b5faa573633c41a7 ===
Gold sentences and facts:
Sentence 0: The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly .
  Fact 0.0: The Roosevelt Corollary was an addition to the Monroe Doctrine.
  Fact 0.1: President Theodore Roosevelt articulated the Roosevelt Corollary.
  Fact 0.2: Theodore Roosevelt articulated the Roosevelt Corollary in his State of the Union address in 1904.
  Fact 0.3: The Roosevelt Corollary was articulated after the Venezuela Crisis of 1902-03.
  Fact 0.4: The Roosevelt Corollary states that the United States will intervene in conflicts between European countries and Latin American countries.
  Fact 0.5: Under the Roosevelt Corollary, the United States intervenes to enforce legitimate claims of the European powers.
  Fact 0.6: Under the Roosevelt Corollary, the United States intervenes rather than having the Europeans press their claims directly.

Retrieved items:
<item 1>
Roosevelt Corollary
Contents ( hide ) 1 Use 1.1 Shift to the `` Good Neighbor '' policy 2 Criticism 3 See also 4 References 5 Bibliography Use ( edit ) See also : Overseas
interventions of the United States The Roosevelt Corollary was an addition to the Monroe Doctrine ; however , it could be seen as a departure . While the Monroe
Doctrine said European countries should stay out of Latin America , the Roosevelt Corollary took this further to say he had the right to exercise military force
in Latin American countries to keep European countries out . Historian Walter LaFeber wrote ( Roosevelt ) essentially turns the Monroe Doctrine on its head and
toes says the Europeans should stay out , but the United States has the right , under the doctrine , to go in to exercise police power to keep the Europeans out
of the way .
</item 1>

<item 2>
American moralists believed the United States should avoid reactionary empires and preserve democratic ideals, trusting the world would eventually embrace them.
In 1917, they feared democracy’s defeat. Religious leaders, women led by Jane Addams, William Jennings Bryan, and especially President Woodrow Wilson supported
moralism. Wilson won their backing by calling for a “war to make the world safe for democracy,” and Americans overwhelmingly agreed to fight for their ideals.
</item 2>

<item 3>
The 1904 Roosevelt Corollary expanded the Monroe Doctrine, claiming the U.S. could act as a regional “police power” to prevent European intervention, enforce
debts, and address instability. Applied in the Dominican Republic, it helped justify occupations and interventions across Latin America and was criticized as
imperialist. The Clark Memorandum, Hoover’s policies, and Franklin Roosevelt’s 1934 Good Neighbor policy marked retreats, though Cold War priorities revived
interventionism. The fragmented material also includes bibliographies
</item 3>

<item 4>
Cuban War of Independence
Joseph E. Wisan wrote in an essay titled `` The Cuban Crisis As Reflected In The New York Press '' , published in `` American Imperialism '' in 1898 : `` In the
opinion of the writer , the Spanish -- American War would not have occurred had not the appearance of Hearst in New York journalism precipitated a bitter battle
for newspaper circulation . '' It has also been argued that the main reason the United States entered the war was its failed attempt to purchase Cuba from Spain
. Further information : Propaganda of the Spanish -- American War Hostilities started hours after the declaration of war when a contingent of US Navy ships
under Admiral William T. Sampson blockaded several Cuban ports . The Americans decided to invade Cuba and to start in Oriente , where the Cubans had almost
absolute control . They cooperated by establishing a beachhead and protecting the U.S. landing in Daiquiri . The first U.S. objective was to capture the city of
Santiago de Cuba in order to destroy Linares ' army and Cervera 's fleet .
</item 4>

<item 5>
The Roosevelt Corollary expanded the Monroe Doctrine: while barring European involvement in Latin America, it claimed the United States could intervene
militarily there as a “police power” to keep Europeans out. Critics view it as an imperialist justification for U.S. intervention and corporate advantage; it
was cited to justify Operation PBSUCCESS, which ousted Guatemala’s elected president Jacobo Árbenz and installed Carlos Castillo Armas. Other critics argue such
policies harm national security and domestic politics.
</item 5>

<item 6>
The United States demanded respect for neutral ships and condemned civilian deaths at sea. Britain seized American vessels, but backed down after Wilson
protested; his adviser Edward M. House said Britain had violated neutral rights extensively, though courteously. German submarines sank ships without warning,
arguing that surfacing was dangerous because British merchant ships were armed and too small to rescue submarine crews. The U.S. warned Germany in February 1915
about submarine warfare; on April 22, the German Embassy issued a warning to Americans
</item 6>

<item 7>
The list traces settlements across the Americas from prehistoric Tlapacoya (7500 BC), Puerto Hormiga (4000 BC), and Caral (2627 BC) through 1732. It includes
Indigenous cities and pueblos, Norse L’Anse aux Meadows (1001), and European-founded settlements, especially Spanish colonial cities established from 1492
onward. Entries span present-day Latin America, the Caribbean, Canada, the United States, and Greenland, noting distinctions such as
</item 7>

<item 8>
Roosevelt Corollary
The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the
Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries
to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly . Roosevelt tied his policy to the Monroe
Doctrine , and it was also consistent with his foreign policy included in his Big Stick Diplomacy . Roosevelt stated that in keeping with the Monroe Doctrine ,
the United States was justified in exercising `` international police power '' to put an end to chronic unrest or wrongdoing in the Western Hemisphere . While
the Monroe Doctrine had sought to prevent European intervention , the Roosevelt Corollary was used to justify US intervention throughout the hemisphere . In
1934 , President Franklin D. Roosevelt renounced interventionism and established his Good Neighbor policy for the Western Hemisphere .
</item 8>

<item 9>
The 1904 Roosevelt Corollary expanded the Monroe Doctrine after the 1902–03 Venezuela Crisis. Theodore Roosevelt claimed the United States could act as an
“international police power,” intervening in Latin American conflicts to enforce European claims and prevent chronic unrest—justifying U.S. intervention across
the hemisphere. In 1934, Franklin D. Roosevelt renounced interventionism in favor of the Good Neighbor policy.
</item 9>

<item 10>
Serge Ricard argues that the Roosevelt Corollary was not simply an extension of the Monroe Doctrine, which opposed European intervention in the Americas.
Instead, it established a new diplomatic principle reflecting Theodore Roosevelt’s “big stick” policy: the United States would assume the interventionist role
previously held by European powers. Related topics include Dollar Diplomacy, the Drago Doctrine, and U.S. imperial expansion.
</item 10>
=== END ITEM id=b5faa573633c41a7 ===

=== ITEM id=b40954454baae224 ===
Gold sentences and facts:
Sentence 0: 2017 World Series Team ( Wins ) Manager ( s ) Season Houston Astros ( 4 ) A.J. Hinch 101 -- 61 (. 623 ) Los Angeles Dodgers ( 3 ) Dave Roberts 104 -- 58 (. 642 ) Dates October 24 -- November 1 MVP George Springer ( Houston ) Umpires Phil Cuzzi ( Games 1 -- 2 ) , Gerry Davis ( crew chief ) , Laz Díaz , Dan Iassogna , Bill Miller , Paul Nauert , Mark Wegner ( Games 3 -- 7 ) ALCS Houston Astros defeated New York Yankees , 4 -- 3 NLCS Los Angeles Dodgers defeated Chicago Cubs , 4 -- 1 Broadcast Television Fox Fox Deportes MLB International TV announcers Joe Buck , John Smoltz , Ken Rosenthal , and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez , and Jaime Motta ( Fox Deportes ) Matt Vasgersian and Buck Martinez ( MLB International ) Radio ESPN Radio ESPN Deportes Radio Sportstalk 790 ( HOU -- English ) FM 94.1 HD3 ( HOU -- Spanish ) AM 570 LA Sports ( LAD -- English ) Univision America 1020 ( LAD -- Spanish ) Radio Korea 1540 ( LAD -- Korean ) Radio announcers Dan Shulman and Aaron Boone ( ESPN ) Eduardo Ortega , Renato Bermúdez , José Francisco Rivera , and Orlando Hernández ( ESPN Deportes ) Robert Ford and Steve Sparks ( HOU -- English ) Alex Treviño and Francisco Romero ( HOU -- Spanish ) Charley Steiner and Rick Monday ( LAD -- English ) Jaime Jarrín and Jorge Jarrín ( LAD -- Spanish ) Richard Choi and Chong Ho Yim ( LAD -- Korean ) World Series Program ← 2016 World Series
  Fact 0.0: The 2017 World Series table lists the Houston Astros as the team with 4 wins.
  Fact 0.1: The 2017 World Series table lists the Los Angeles Dodgers as the team with 3 wins.
  Fact 0.2: A.J. Hinch managed the Houston Astros in the 2017 season, with a record of 101 wins and 61 losses (.623).
  Fact 0.3: Dave Roberts managed the Los Angeles Dodgers in the 2017 season, with a record of 104 wins and 58 losses (.642).
  Fact 0.4: The 2017 World Series was played from October 24 to November 1.
  Fact 0.5: George Springer of the Houston Astros was the MVP of the 2017 World Series.
  Fact 0.6: Phil Cuzzi was an umpire for Games 1 and 2 of the 2017 World Series.
  Fact 0.7: Gerry Davis was the crew chief umpire of the 2017 World Series.
  Fact 0.8: Laz Diaz was an umpire of the 2017 World Series.
  Fact 0.9: Dan Iassogna was an umpire of the 2017 World Series.
  Fact 0.10: Bill Miller was an umpire of the 2017 World Series.
  Fact 0.11: Paul Nauert was an umpire of the 2017 World Series.
  Fact 0.12: Mark Wegner was an umpire for Games 3 to 7 of the 2017 World Series.
  Fact 0.13: The Houston Astros defeated the New York Yankees 4 to 3 in the 2017 ALCS.
  Fact 0.14: The Los Angeles Dodgers defeated the Chicago Cubs 4 to 1 in the 2017 NLCS.
  Fact 0.15: The 2017 World Series was broadcast on Fox television.
  Fact 0.16: The 2017 World Series was broadcast on Fox Deportes.
  Fact 0.17: The 2017 World Series was broadcast on MLB International TV.
  Fact 0.18: Joe Buck, John Smoltz, Ken Rosenthal and Tom Verducci were TV announcers on Fox.
  Fact 0.19: Rolando Nichols, Carlos Alvarez, Edgar Gonzalez and Jaime Motta were TV announcers on Fox Deportes.
  Fact 0.20: Matt Vasgersian and Buck Martinez were TV announcers on MLB International.
  Fact 0.21: The 2017 World Series was broadcast on ESPN Radio.
  Fact 0.22: The 2017 World Series was broadcast on ESPN Deportes Radio.
  Fact 0.23: Sportstalk 790 carried the Houston Astros English radio broadcast.
  Fact 0.24: FM 94.1 HD3 carried the Houston Astros Spanish radio broadcast.
  Fact 0.25: AM 570 LA Sports carried the Los Angeles Dodgers English radio broadcast.
  Fact 0.26: Univision America 1020 carried the Los Angeles Dodgers Spanish radio broadcast.
  Fact 0.27: Radio Korea 1540 carried the Los Angeles Dodgers Korean radio broadcast.
  Fact 0.28: Dan Shulman and Aaron Boone were radio announcers on ESPN.
  Fact 0.29: Eduardo Ortega, Renato Bermudez, Jose Francisco Rivera and Orlando Hernandez were radio announcers on ESPN Deportes.
  Fact 0.30: Robert Ford and Steve Sparks were English radio announcers for the Houston Astros.
  Fact 0.31: Alex Trevino and Francisco Romero were Spanish radio announcers for the Houston Astros.
  Fact 0.32: Charley Steiner and Rick Monday were English radio announcers for the Los Angeles Dodgers.
  Fact 0.33: Jaime Jarrin and Jorge Jarrin were Spanish radio announcers for the Los Angeles Dodgers.
  Fact 0.34: Richard Choi and Chong Ho Yim were Korean radio announcers for the Los Angeles Dodgers.
  Fact 0.35: The 2017 World Series followed the 2016 World Series.
Sentence 1: The 2017 World Series was the championship series of Major League Baseball 's ( MLB ) 2017 season . The 113th edition of the World Series , it was played between October 24 and November 1 . The series was a best - of - seven playoff between the National League ( NL ) champion Los Angeles Dodgers and the American League ( AL ) champion Houston Astros . It was sponsored by the internet television service YouTube TV and officially known as the World Series presented by YouTube TV .
  Fact 1.0: The 2017 World Series was the championship series of Major League Baseball's 2017 season.
  Fact 1.1: The 2017 World Series was the 113th edition of the World Series.
  Fact 1.2: The 2017 World Series was played between October 24 and November 1.
  Fact 1.3: The 2017 World Series was a best-of-seven playoff.
  Fact 1.4: The 2017 World Series was played between the National League champion Los Angeles Dodgers.
  Fact 1.5: The 2017 World Series was played between the American League champion Houston Astros.
  Fact 1.6: The 2017 World Series was sponsored by the internet television service YouTube TV.
  Fact 1.7: The 2017 World Series was officially known as the World Series presented by YouTube TV.

Retrieved items:
<item 1>
2005 World Series
1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006
2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most Valuable Players Starting pitchers Babe Ruth
Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7
1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World Series Chicago White Sox Based in Chicago ,
</item 1>

<item 2>
Game seven
1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999
2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most
Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV
ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World
Series Stanley Cup Finals Stanley Cup awarded on challenge basis 1893 --
</item 2>

<item 3>
2017 World Series
Sports Business Journal . Retrieved October 4 , 2017 . External links ( edit ) 2017 World Series at Baseball Reference 2017 Major League Baseball postseason
Part of the 2017 Major League Baseball season World Series American League Championship Series National League Championship Series American League Division
Series National League Division Series American League Wild Card Game National League Wild Card Game American League teams Boston Red Sox Cleveland Indians
Houston Astros Minnesota Twins New York Yankees National League teams Arizona Diamondbacks Chicago Cubs Colorado Rockies Los Angeles Dodgers Washington
Nationals World Series No World Series was held in 1904 because the NL champions refused to participate ; no World Series was held in 1994 due to a players '
strike . 1900s -- 1910s 1900 1901 1902 1903 1904 1905 1906 1907 1908 1909 1910 1911 1912 1913 1914 1915 1916 1917 1918 1919 1920s -- 1930s 1920 1921 1922 1923
1924 1925 1926 1927 1928 1929
</item 3>

<item 4>
2017 World Series
The 113th edition of the World Series , it was played between October 24 and November 1 . The series was a best - of - seven playoff between the National League
( NL ) champion Los Angeles Dodgers and the American League ( AL ) champion Houston Astros . It was sponsored by the internet television service YouTube TV and
officially known as the World Series presented by YouTube TV . The Astros defeated the Dodgers , four games to three , to win their first World Series in
franchise history , and the first team from Texas to do so . Both teams set a World Series record with a combined total of 25 home runs throughout the entire
series , including a team record 15 home runs by the Astros , and hit a combined total of eight home runs in Game 2 to set the single game World Series mark .
Houston outfielder George Springer was named as the World Series Most Valuable Player ( MVP ) after hitting five home runs in the series to tie a World Series
record with Reggie Jackson in 1977 and Dodgers ' 2017 World Series representative Chase Utley in 2009 when he played for the Philadelphia Phillies .
</item 4>

<item 5>
2017 World Series
1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series
champions Most Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise )
Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series
Category : World Series Fox Major League Baseball Related programs The Cheap Seats ( 2010 -- 2011 ) MLB Whiparound Major League Baseball Game of the Week
Thursday Night Baseball ( 1997 -- 2001 ) This Week in Baseball ( 2000 -- 2011 ) Related articles DirecTV N3D FoxBox FoxTrax Scooter Television contracts ( cable
) MLB Network World Series television ratings National coverage Fox ( 1996 -- present ) FS1 ( 2014 -- present
</item 5>

<item 6>
2017 World Series
October 25 , 2017 , 5 : 17 pm ( PDT ) at Dodger Stadium in Los Angeles , California , 93 ° F ( 34 ° C ) , clear Team 5 6 7 8 9 10 11 Houston 0 0 0 0 0 0 7 14
Los Angeles 0 0 0 0 0 0 0 6 5 0 WP : Chris Devenski ( 1 -- 0 ) LP : Brandon McCarthy ( 0 -- 1 ) Home runs : HOU : Marwin González ( 1 ) , José Altuve ( 1 ) ,
Carlos Correa ( 1 ) , George Springer ( 1 ) LAD : Joc Pederson ( 1 ) , Corey Seager ( 1 ) , Yasiel Puig ( 1 ) , Charlie Culberson ( 1 ) Attendance : 54,293 The
starting pitchers for Game 2 were Rich Hill for the Dodgers and Justin Verlander for the Astros . The Astros scored first when Bregman drove in Josh Reddick
with a hit in the third inning .
</item 6>

<item 7>
2017 World Series
October 28 , 2017 , 7 : 20 pm ( CDT ) at Minute Maid Park in Houston , Texas , 67 ° F ( 19 ° C ) , roof closed Team 5 6 7 8 9 Los Angeles 0 0 0 0 0 0 0 5 6 7 0
Houston 0 0 0 0 0 0 0 0 WP : Tony Watson ( 1 -- 0 ) LP : Ken Giles ( 0 -- 1 ) Home runs : LAD : Joc Pederson ( 2 ) HOU : George Springer ( 2 ) , Alex Bregman (
2 ) Attendance : 43,322 The starting pitchers for Game 4 were Alex Wood for the Dodgers and Charlie Morton for the Astros . Springer homered off Wood in the
bottom of the sixth for the first run of the ballgame . It was the only hit Wood allowed in ​ 5 ⁄ innings pitched in the game . Forsythe drove in Cody Bellinger
to tie the game in the top of the seventh .
</item 7>

<item 8>
2017 World Series
`` Overview of World Series home - field advantage '' . Major League Baseball . Retrieved November 2 , 2017 . We are just a week away from the World Series
presented by YouTube TV Jump up ^ `` Astros , Dodgers complete record - setting World Series '' . ESPN . November 2 , 2017 . Retrieved November 2 , 2017 . ^
Jump up to : Castrovince , Anthony ( November 1 , 2017 ) . `` Springer swats his way to MVP honors '' . Major League Baseball . Retrieved November 2 , 2017 .
Jump up ^ Schoenfield , David ( November 2 , 2017 ) . `` George Springer wins World Series MVP , ties record with five homers '' . ESPN . Retrieved November 2 ,
2017 . Jump up ^ Whicker , Mark ( October 22 , 2017 ) . `` Whicker : Dodgers , Houston Astros try to rekindle their 80s rivalry '' . Orange County Register .
Retrieved October 25 , 2017 .
</item 8>

<item 9>
2017 World Series
and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez , and Jaime Motta ( Fox Deportes ) Matt Vasgersian and Buck Martinez ( MLB
International ) Radio ESPN Radio ESPN Deportes Radio Sportstalk 790 ( HOU -- English ) FM 94.1 HD3 ( HOU -- Spanish ) AM 570 LA Sports ( LAD -- English )
Univision America 1020 ( LAD -- Spanish ) Radio Korea 1540 ( LAD -- Korean ) Radio announcers Dan Shulman and Aaron Boone ( ESPN ) Eduardo Ortega , Renato
Bermúdez , José Francisco Rivera , and Orlando Hernández ( ESPN Deportes ) Robert Ford and Steve Sparks ( HOU -- English ) Alex Treviño and Francisco Romero (
HOU -- Spanish ) Charley Steiner and Rick Monday ( LAD -- English ) Jaime Jarrín and Jorge Jarrín ( LAD -- Spanish ) Richard Choi and Chong Ho Yim ( LAD --
Korean ) World Series Program ← 2016 World Series The 2017 World Series was the championship series of Major League Baseball 's ( MLB ) 2017 season .
</item 9>

<item 10>
2017 World Series
2017 World Series - wikipedia 2017 World Series Jump to : navigation , search 2017 World Series Team ( Wins ) Manager ( s ) Season Houston Astros ( 4 ) A.J.
Hinch 101 -- 61 (. 623 ) Los Angeles Dodgers ( 3 ) Dave Roberts 104 -- 58 (. 642 ) Dates October 24 -- November 1 MVP George Springer ( Houston ) Umpires Phil
Cuzzi ( Games 1 -- 2 ) , Gerry Davis ( crew chief ) , Laz Díaz , Dan Iassogna , Bill Miller , Paul Nauert , Mark Wegner ( Games 3 -- 7 ) ALCS Houston Astros
defeated New York Yankees , 4 -- 3 NLCS Los Angeles Dodgers defeated Chicago Cubs , 4 -- 1 Broadcast Television Fox Fox Deportes MLB International TV announcers
Joe Buck , John Smoltz , Ken Rosenthal , and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez ,
</item 10>
=== END ITEM id=b40954454baae224 ===

=== ITEM id=0fd4400468e75bd4 ===
Gold sentences and facts:
Sentence 0: Nigeria is the largest oil and gas producer in Africa . Crude oil from the delta basin comes in two types : light , and comparatively heavy -- the lighter around 36 gravity and the heavier , 20 -- 25 gravity . Both types are paraffinic and low in sulfur .
  Fact 0.0: Nigeria is the largest oil and gas producer in Africa.
  Fact 0.1: Crude oil from the delta basin comes in two types: light, and comparatively heavy.
  Fact 0.2: The lighter type has a gravity of around 36.
  Fact 0.3: The heavier type has a gravity of 20 to 25.
  Fact 0.4: Both types of crude oil from the delta basin are paraffinic.
  Fact 0.5: Both types of crude oil from the delta basin are low in sulfur.

Retrieved items:
<item 1>
Sahara
The desert comprises much of North Africa , excluding the fertile region on the Mediterranean Sea coast , the Atlas Mountains of the Maghreb , and the Nile
Valley in Egypt and Sudan . It stretches from the Red Sea in the east and the Mediterranean in the north to the Atlantic Ocean in the west , where the landscape
gradually changes from desert to coastal plains . To the south , it is bounded by the Sahel , a belt of semi-arid tropical savanna around the Niger River valley
and the Sudan Region of Sub-Saharan Africa . The Sahara can be divided into several regions including : the western Sahara , the central Ahaggar Mountains , the
Tibesti Mountains , the Aïr Mountains , the Ténéré desert , and the Libyan Desert . For several hundred thousand years , the Sahara has alternated between
desert and savanna grassland in a 41,000 year cycle caused by changes ( `` precession '' ) in the Earth 's axis as it rotates around the sun , which change the
location of the North African Monsoon . It is next expected to become green in about 15,000 years ( 17,000 AD ) .
</item 1>

<item 2>
Sub-Saharan Africa
According to the World Bank , the economic growth rate in the region had risen to 4.7 % in 2013 , with a rate of 5.2 % forecasted for 2014 . This continued rise
was attributed to increasing investment in infrastructure and resources as well as steady expenditure per household . Energy and power ( edit ) Main article :
Mineral industry of Africa Oil production by country ( with other key actors of African or oil economy ) Rank Area bb / day Year Like ... _ W : World 85540000
2007 est . 01 E : Russia 9980000 2007 est . 02 Ar : Saudi Arb 9200000 2008 est . 04 As : Libya 4725000 2008 est . Iran 10 Af : Nigeria / Africa 2352000 2011 est
. Norway 15 Af : Algeria 2173000 2007 est . 16 Af : Angola 1910000 2008 est . 17 Af : Egypt 1845000 2007 est . 27 Af : Tunisia 664000 2007 est . Australia 31 Af
: Sudan 466100 2007 est .
</item 2>

<item 3>
Sub-Saharan Africa
Australia 31 Af : Sudan 466100 2007 est . Ecuador 33 Af : Eq. Guinea 368500 2007 est . Vietnam 38 Af : DR Congo 261000 2008 est . 39 Af : Gabon 243900 2007 est
. 40 Af : Sth Africa 199100 2007 est . 45 Af : Chad 156000 2008 est . Germany 53 Af : Cameroon 87400 2008 est . France 56 E : France 71400 2007 60 Af : Ivory
Coast 54400 2008 est . _ Af : Africa 10780400 2011 Russia Source : CIA.gov , World Facts Book > Oil exporters . As of 2009 , fifty percent of Africa is rural
with no access to electricity . Africa generates 47 GW of electricity , less than 0.6 % of the global market share . Many countries are affected by power
shortages . Because of rising prices in commodities such as coal and oil , thermal sources of energy are proving to be too expensive for power generation .
</item 3>

<item 4>
Sub-Saharan Africa
Transportation cost is low and no pipelines have to be laid as in Central Asia . Almost all reserves are offshore , so political turmoil within the host country
will not directly interfere with operations . Sub-Saharan oil is viscous , with a very low sulfur content . This quickens the refining process and effectively
reduces costs . New sources of oil are being located in Sub-Saharan Africa more frequently than anywhere else . Of all new sources of oil , 1⁄3 are in
Sub-Saharan Africa . Agriculture ( edit ) Agricultural fields in Rwanda 's Eastern Province Sub-Saharan Africa has more variety of grains than anywhere in the
world . Between 13,000 and 11,000 BCE wild grains began to be collected as a source of food in the cataract region of the Nile , south of Egypt . The collecting
of wild grains as source of food spread to Syria , parts of Turkey , and Iran by the eleventh millennium BCE . By the tenth and ninth millennia southwest Asians
domesticated their wild grains , wheat , and barley after the notion of collecting wild grains spread from the Nile . Numerous crops have been domesticated in
the region and spread to other parts of the world .
</item 4>

<item 5>
Sub-Saharan Africa
Barbary Coast Bashmur Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba
Mountains Nubia The Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf
of Guinea Niger Basin Guinean Forests of West Africa Niger Delta Inner Niger Delta Southern Africa Madagascar Central Highlands ( Madagascar ) Northern
Highlands Rhodesia North South Thembuland Succulent Karoo Nama Karoo Bushveld Highveld Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay
Hydra Bay Macro-regions Aethiopia Arab world Commonwealth realm East African montane forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater
Middle East Islands of Africa List of countries where Arabic is an official language Mediterranean Basin MENA MENASA Middle East Mittelafrika Negroland
Northeast Africa Portuguese - speaking African countries Sahara Sahel Sub-Saharan Africa Sudan ( region ) Sudanian Savanna Tibesti Mountains Tropical Africa
Regions of Asia Central Greater Middle East Aral Sea Aralkum Desert Caspian Sea Dead Sea Sea of Galilee Transoxiana Turan Greater Khorasan
</item 5>

<item 6>
Petroleum industry in Nigeria
An open - air market for illegal crude oil operates off the Niger Delta , called the Togo Triangle . Natural gas ( edit ) Natural gas reserves are well over
5,300 km ( 187 × 10 ^ cu ft ) , the gas reserves are three times as substantial as the crude oil reserves . The biggest natural gas initiative is the Nigerian
Liquified Natural Gas Company , which is operated jointly by several companies and the state . It began exploration and production in 1999 . Chevron is also
attempting to create the Escravos Gas Utilization project which will be capable of producing 4,500,000 m ( 160 × 10 ^ cu ft ) per day . gas reserves . In 2008 ,
the government prepared a Gas Master Plan that was intended to promote natural gas production and encourage the supply of natural gas to domestic power stations
so as to help alleviate the country 's electricity shortages . There is also an export gas pipeline , known as the West African Gas Pipeline , in the works but
has encountered numerous setbacks . The pipeline would allow for transportation of natural gas to Benin , Ghana , Togo , and Cote d'Ivoire .
</item 6>

<item 7>
Sahara
Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba Mountains Nubia The
Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf of Guinea Niger Basin
Guinean Forests of West Africa Niger Delta Inner Niger Delta Southern Africa Madagascar Central Highlands ( Madagascar ) Northern Highlands Rhodesia North South
Thembuland Succulent Karoo Nama Karoo Bushveld Highveld Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay Hydra Bay Macro-regions Aethiopia
Arab world Commonwealth realm East African montane forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater Middle East Islands of Africa
List of countries where Arabic is an official language Mediterranean Basin MENA MENASA Middle East Mittelafrika Negroland Northeast Africa Portuguese - speaking
African countries Sahara Sahel Sub-Saharan Africa Sudan ( region ) Sudanian Savanna Tibesti Mountains Tropical Africa Coordinates : 23 ° N 13 ° E ﻿ / ﻿ 23 ° N
13 ° E ﻿ / 23 ;
</item 7>

<item 8>
OPEC
Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa
1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 --
1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa
1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,
</item 8>

<item 9>
Petroleum industry in Nigeria
Petroleum industry in Nigeria - wikipedia Petroleum industry in Nigeria Jump to : navigation , search Nigeria is the largest oil and gas producer in Africa .
Crude oil from the delta basin comes in two types : light , and comparatively heavy -- the lighter around 36 gravity and the heavier , 20 -- 25 gravity . Both
types are paraffinic and low in sulfur . The gates of the oil refinery in Port Harcourt Contents ( hide ) 1 History of oil exploration 2 Production and
exploration 2.1 Offshore 2.2 Natural gas 2.3 Downstream 3 Current manufacturing 4 History and politics 4.1 Colonial legacy ( 1800s -- 1960s ) 4.2 Implications
and causes of civil war ( 1966 -- 1970 ) 4.3 Industry nationalisation ( 1970 -- 1979 ) 4.4 Attempted democracy and debt ( 1979 -- 1983 ) 4.5 Return to military
rule and electoral annulment ( 1983 -- 1993 ) 4.6 An environment of crisis ( 1993 -- present ) 5 Operating agreements 5.
</item 9>

<item 10>
Sub-Saharan Africa
Major suppliers are South Africa , Ghana , Zimbabwe , Tanzania , Guinea , and Mali . South Africa had been first in the world in terms of gold production since
1905 , but in 2007 it moved to second place , according to GFMS , the precious metals consultancy . Uranium is major commodity from the region . Significant
suppliers are Niger , Namibia , and South Africa . Namibia was the number one supplier from Sub-Saharan Africa in 2008 . The region produces 49 % of the world
's diamonds . By 2015 , it is estimated that 25 % of North American oil will be from Sub-Saharan Africa , ahead of the Middle East . Sub-Saharan Africa has been
the focus of an intense race for oil by the West , China , India , and other emerging economies , even though it holds only 10 % of proven oil reserves , less
than the Middle East . This race has been referred to as the second Scramble for Africa . All reasons for this global scramble come from the reserves ' economic
benefits . Transportation cost is low and no pipelines have to be laid as in Central Asia .
</item 10>
=== END ITEM id=0fd4400468e75bd4 ===

=== ITEM id=44077f7bfe0fa4ee ===
Gold sentences and facts:
Sentence 0: The Uralic languages ( / jʊəˈrælɪk / ; sometimes called Uralian languages / jʊəˈreɪliən / ) form a language family of 38 languages spoken by approximately 25 million people , predominantly in Northern Eurasia . The Uralic languages with the most native speakers are Hungarian , Finnish , and Estonian , which are the official languages of Hungary , Finland , and Estonia , respectively , and of the European Union . Other Uralic languages with significant numbers of speakers are Erzya , Moksha , Mari , Udmurt , and Komi , which are officially recognized languages in various regions of Russia .
  Fact 0.0: The Uralic languages form a language family.
  Fact 0.1: The Uralic languages are sometimes called Uralian languages.
  Fact 0.2: The Uralic language family has 38 languages.
  Fact 0.3: The Uralic languages are spoken by approximately 25 million people.
  Fact 0.4: The Uralic languages are spoken predominantly in Northern Eurasia.
  Fact 0.5: Hungarian, Finnish, and Estonian are the Uralic languages with the most native speakers.
  Fact 0.6: Hungarian is the official language of Hungary.
  Fact 0.7: Finnish is the official language of Finland.
  Fact 0.8: Estonian is the official language of Estonia.
  Fact 0.9: Hungarian, Finnish, and Estonian are official languages of the European Union.
  Fact 0.10: Erzya, Moksha, Mari, Udmurt, and Komi are other Uralic languages with significant numbers of speakers.
  Fact 0.11: Erzya, Moksha, Mari, Udmurt, and Komi are officially recognized languages in various regions of Russia.

Retrieved items:
<item 1>
Uralic languages
Nonetheless , these relationships were not widely accepted . Hungarian intellectuals especially were not interested in the theory and preferred to assume
connections with Turkic tribes , an attitude characterized by Merritt Ruhlen as due to `` the wild unfettered Romanticism of the epoch '' . Still , in spite of
this hostile climate , the Hungarian Jesuit János Sajnovics travelled with Maximilian Hell to survey the alleged relationship between Hungarian and Sami .
Sajnovics published his results in 1770 , arguing for a relationship based on several grammatical features . In 1799 , the Hungarian Sámuel Gyarmathi published
the most complete work on Finno - Ugric to that date . Uralic languages in the Russian Empire ( Russian Census of 1897 ; the census was not held in Finland
because it was an autonomous area ) Progress resumed after a number of decades , with the first major field research expeditions on the Uralic languages spoken
in the more eastern parts of Russia .
</item 1>

<item 2>
Uralic languages
Classification ( edit ) Relative numbers of speakers of Uralic languages Hungarian 56 % Finnish 20 % Estonian 4.2 % Erzya 2.8 % Moksha 2.5 % Mari 2 % Udmurt 1.9
% Komi 1.6 % Other 8.9 % The Uralic family comprises nine undisputed groups with no consensus classification between them . ( Some of the proposals are listed
in the next section . ) An agnostic approach treats them as separate branches . Obsolete or native names are displayed in italics .
</item 2>

<item 3>
Uralic languages
As they became better known in the early 20th century , they were found to be quite divergent , and they were assumed to have separated already early on . The
terminology adopted for this was `` Uralic '' for the entire family , `` Finno - Ugric '' for the non-Samoyedic languages ( though `` Finno - Ugric '' has , to
this day , remained in use also as a synonym for the whole family ) . Finno - Ugric and Samoyedic are listed in ISO 639 - 5 as primary branches of Uralic .
Nodes of the traditional family tree recognized in some overview sources : Year Author ( s ) Finno - Ugric Ugric Ob - Ugric Finno - Permic Finno - Volgaic Volga
- Finnic Finno - Samic 1910 Szinnyei ✓ ✓ ✓ ✓ ✓ ✗ ✓ 1921 T.I.
</item 3>

<item 4>
Uralic languages
The Uralic languages with the most native speakers are Hungarian , Finnish , and Estonian , which are the official languages of Hungary , Finland , and Estonia
, respectively , and of the European Union . Other Uralic languages with significant numbers of speakers are Erzya , Moksha , Mari , Udmurt , and Komi , which
are officially recognized languages in various regions of Russia . The name `` Uralic '' derives from the fact that the areas where the languages are spoken are
found on both sides of the Ural Mountains . Also , the original homeland ( Urheimat ) is commonly hypothesized to be in the vicinity of the Urals . Finno -
Ugric is sometimes used as a synonym for Uralic , though Finno - Ugric is widely understood to exclude the Samoyedic languages . Scholars who do not accept the
traditional notion that Samoyedic split first from the rest of the Uralic family may treat the terms as synonymous .
</item 4>

<item 5>
Uralic languages
Uralic languages - wikipedia Uralic languages Jump to : navigation , search Not to be confused with Urali language . This article needs additional citations for
verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( July 2017 ) (
Learn how and when to remove this template message ) Uralic Uralian languages Geographic distribution Central , Eastern , and Northern Europe , North Asia
Linguistic classification One of the world 's primary language families Proto - language Proto - Uralic Subdivisions Finnic Hungarian Khanty Mansi Mari
Mordvinic Permic Sami Samoyedic ISO 639 - 5 urj Glottolog ural1272 Geographical distribution of the Uralic languages The Uralic languages ( / jʊəˈrælɪk / ;
sometimes called Uralian languages / jʊəˈreɪliən / ) form a language family of 38 languages spoken by approximately 25 million people , predominantly in
Northern Eurasia .
</item 5>

<item 6>
Uralic languages
The Finno - Permic grouping still holds some support , though the arrangement of its subgroups is a matter of some dispute . Mordvinic is commonly seen as
particularly closely related to or part of Finno - Samic . The term Volgaic ( or Volga - Finnic ) was used to denote a branch previously believed to include
Mari , Mordvinic and a number of the extinct languages , but it is now obsolete and considered a geographic classification rather than a linguistic one . Within
Ugric , uniting Mansi with Hungarian rather than Khanty has been a competing hypothesis to Ob - Ugric . Lexical isoglosses ( edit ) Lexicostatistics has been
used in defense of the traditional family tree . A recent re-evaluation of the evidence however fails to find support for Finno - Ugric and Ugric , suggesting
four lexically distinct branches ( Finno - Permic , Hungarian , Ob - Ugric and Samoyedic ) . One alternate proposal for a family tree , with emphasis on the
development of numerals , is as follows : Uralic ( * kektä `` 2 '' ,
</item 6>

<item 7>
Uralic languages
A number of umlaut processes are found in Livonian . In Komi , but not in Udmurt . Possible Relations with other families ( edit ) Many relationships between
Uralic and other language families have been suggested , but none of these are generally accepted by linguists at the present time . Indo - Uralic ( edit ) Main
article : Indo - Uralic languages The Indo - Uralic ( or Uralo - Indo - European ) hypothesis suggests that Uralic and Indo - European are related at a fairly
close level or , in its stronger form , that they are more closely related than either is to any other language family . Uralic -- Yukaghir ( edit ) Main
article : Uralic -- Yukaghir languages The Uralic -- Yukaghir hypothesis identifies Uralic and Yukaghir as independent members of a single language family . It
is currently widely accepted that the similarities between Uralic and Yukaghir languages are due to ancient contacts . Regardless , the hypothesis is accepted
by a few linguists and viewed as attractive by a somewhat larger number .
</item 7>

<item 8>
Uralic languages
Obsolete or native names are displayed in italics . Finnic ( Fennic , Baltic Finnic , Balto - Finnic , Balto - Fennic ) Hungarian ( Magyar ) Khanty ( Ostyak ,
Handi , Hantõ ) Mansi ( Vogul ) Mari ( Cheremis ) Mordvinic ( Mordvin , Mordvinian ) Permic ( Permian ) Sami ( Saami , Samic , Saamic , Lappic , Lappish )
Samoyedic ( Samoyed ) There is also historical evidence of a number of extinct languages of uncertain affiliation : Merya Muromian Meshcherian ( until 16th
century ? ) Traces of Finno - Ugric substrata , especially in toponymy , in the northern part of European Russia have been proposed as evidence for even more
extinct Uralic languages . Traditional classification ( edit ) All Uralic languages are thought to have descended , through independent processes of language
change , from Proto - Uralic . The internal structure of the Uralic family has been debated since the family was first proposed .
</item 8>

<item 9>
Uralic languages
All of these hypotheses are minority views at the present time in Uralic studies . Other comparisons ( edit ) See also : Alternative theories of the Hungarian
language relations Various unorthodox comparisons have been advanced such as Finno - Basque , Hungarian - Etruscan , and Cal - Ugrian . These are considered at
best spurious fringe - theories by specialists . See also ( edit ) Finland portal Hungary portal Languages portal Uralic peoples Finno - Ugric peoples Notes (
edit ) Jump up ^ Hammarström , Harald ; Forkel , Robert ; Haspelmath , Martin , eds. ( 2017 ) . `` Uralic '' . Glottolog 3.0 . Jena , Germany : Max Planck
Institute for the Science of Human History . Jump up ^ Simons , Gary F. ; Fenning , Charles F. `` Uralic '' . Ethnologue . SIL International . Retrieved 14
February 2018 . Jump up ^ Tommola , Hannu ( 2010 ) . `` Finnish among the Finno - Ugrian languages '' . Mood in the Languages of Europe .
</item 9>

<item 10>
Uralic languages
The internal structure of the Uralic family has been debated since the family was first proposed . Doubts about the validity of most of the proposed higher -
order branchings ( grouping the nine undisputed families ) are becoming more common . A traditional classification of the Uralic languages has existed since the
late 19th century . It has enjoyed frequent adaptation in whole or in part in encyclopedias , handbooks , and overviews of the Uralic family . Donner 's model
is as follows : Ugric ( Ugrian ) Hungarian Ob - Ugric ( Ob Ugrian ) Khanty Mansi Finno - Permic ( Permian - Finnic ) Permic Finno - Volgaic ( Finno - Cheremisic
, Finno - Mari ) Volga - Finnic Mari Mordvinic Finno - Lappic ( Finno - Saamic , Finno - Samic ) Sami Finnic At Donner 's time , the Samoyedic languages were
still poorly known , and he was not able to address their position .
</item 10>
=== END ITEM id=44077f7bfe0fa4ee ===

=== ITEM id=0e72630704aac3a4 ===
Gold sentences and facts:
Sentence 0: `` The Joker '' Cover of the 1973 single Single by Steve Miller Band from the album The Joker B - side `` ' see track listings ' '' Released October 1973 Format 7 '' Single , 12 '' single , Cassette single ( Original Releases ) , CD single and Digital download ( Post Releases ) . Genre Rock , blues rock , southern rock Length 4 : 26 ( Album version ) ; 3 : 35 ( 7 '' version / Radio edit ) and 6 : 20 ( Extended version ) Label Capitol Records ; EMI - Odeon ; London Records , Parlophone Records , EMI Bovema , Odeon Records . Songwriter ( s ) Eddie Curtis , Ahmet Ertegün , Steve Miller Producer ( s ) Steve Miller Steve Miller Band singles chronology `` Fandango '' ( 1972 ) `` The Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash '' ( 1974 ) `` Fandango '' ( 1972 ) `` The Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash '' ( 1974 ) `` Take the Money and Run '' / `` The Joker '' ( 1983 , double A-side , live ) Cover of the 1983 live single Steve Miller Band singles chronology `` Living in the U.S.A. '' ( live ) ( 1983 ) String Module Error : Match not found1983 `` Take the Money and Run '' ( live ) / `` The Joker '' ( live ) ( 1983 ) String Module Error : Match not found1983 `` Shangri - La '' ( 1984 ) Shangri - La1984 `` The Joker '' ( 1990 , reissue ) Cover of the 1990 reissue Steve Miller Band singles chronology `` Ya Ya '' ( 1988 ) Ya Ya1988 `` The Joker '' ( 1990 ) The Joker1990 `` Wide River '' ( 1993 ) Wide River1993
  Fact 0.0: The Joker's 1973 single was released in October 1973 by Steve Miller Band.
  Fact 0.1: The 1973 single 'The Joker' is from the album The Joker.
  Fact 0.2: The Joker single was released in 7-inch single, 12-inch single, and cassette single formats (original releases).
  Fact 0.3: The Joker single was released as a CD single and digital download (later releases).
  Fact 0.4: The Joker single's genres are rock, blues rock, and southern rock.
  Fact 0.5: The Joker single's album version is 4:26 long.
  Fact 0.6: The Joker single's 7-inch version / radio edit is 3:35 long.
  Fact 0.7: The Joker single's extended version is 6:20 long.
  Fact 0.8: The Joker single's label is Capitol Records.
  Fact 0.9: The Joker single was also issued on EMI-Odeon, London Records, Parlophone Records, EMI Bovema, and Odeon Records labels.
  Fact 0.10: The Joker single's songwriters are Eddie Curtis, Ahmet Ertegün, and Steve Miller.
  Fact 0.11: The Joker single's producer is Steve Miller.
  Fact 0.12: The Joker single is a cover of a 1973 single.
  Fact 0.13: Steve Miller Band singles chronology includes 'Fandango' (1972), 'The Joker' (1973), and 'Your Cash Ain't Nothin' But Trash' (1974).
  Fact 0.14: 'Take the Money and Run' / 'The Joker' (1983) was a double A-side live single by Steve Miller Band.
  Fact 0.15: 'The Joker' (1983) live single is a cover of the 1983 live single.
  Fact 0.16: Steve Miller Band singles chronology includes 'Living in the U.S.A.' (live) (1983).
  Fact 0.17: Steve Miller Band singles chronology includes 'Take the Money and Run' (live) / 'The Joker' (live) (1983).
  Fact 0.18: Steve Miller Band singles chronology includes 'Shangri-La' (1984).
  Fact 0.19: Steve Miller Band singles chronology includes 'The Joker' (1990, reissue), a cover of the 1990 reissue.
  Fact 0.20: Steve Miller Band singles chronology includes 'Ya Ya' (1988).
  Fact 0.21: Steve Miller Band singles chronology includes 'The Joker' (1990).
  Fact 0.22: Steve Miller Band singles chronology includes 'Wide River' (1993).
Sentence 1: `` The Joker '' is a song by the Steve Miller Band from their 1973 album The Joker . It is one of two Steve Miller Band songs that feature the nonce word `` pompatus '' . The song topped the US Billboard Hot 100 in early 1974 .
  Fact 1.0: 'The Joker' is a song by the Steve Miller Band.
  Fact 1.1: 'The Joker' is from the Steve Miller Band's 1973 album The Joker.
  Fact 1.2: 'The Joker' is one of two Steve Miller Band songs that feature the nonce word 'pompatus'.
  Fact 1.3: 'The Joker' topped the US Billboard Hot 100 in early 1974.

Retrieved items:
<item 1>
The Joker (Steve Miller Band song)
-- 3 : 59 Personnel ( edit ) Steve Miller -- guitar , lead vocals Gerald Johnson -- bass , backing vocals Dick Thompson -- organ John King -- drums
Certifications ( edit ) Country Certification Date Sales certified Sweden Gold February 6 , 1991 25,000 United Kingdom Silver October 1 , 1990 200,000 United
States Gold January 11 , 1974 1,000,000 Charts ( edit ) Weekly charts ( edit ) Chart ( 1973 / 1974 ) Peak position US Billboard Hot 100 Canadian RPM Top Singles
Dutch Daverende 30 18 Dutch Top 40 18 Australia Kent Music Report 8 Chart ( 1990 / 1991 ) Peak position Austrian Singles Chart 5 Dutch Nationale Top 100 Dutch
Top 40 French SNEP Singles Chart 33 Irish Singles Chart New Zealand RIANZ Singles Chart Norwegian Singles Chart Swedish Singles Chart Swiss Singles Chart 5 UK
Singles Chart Year - end charts ( edit ) Chart ( 1991 ) Position New Zealand ( Recorded Music NZ ) 20 Preceded by `` Time in a Bottle '' by Jim Croce Billboard
Hot 100 number one single January 12 ,
</item 1>

<item 2>
The Joker (Steve Miller Band song)
Formats and track listings ( edit ) 7 '' single ( 1973 ) `` The Joker '' -- 3 : 36 `` Something to Believe In '' -- 4 : 40 7 '' single ( 1983 - live version )
`` The Joker '' ( live ) -- 2 : 55 `` Take the Money and Run '' ( live ) -- 3 : 49 7 '' single ( 1990 ) `` The Joker '' ( single version ) -- 3 : 34 `` Do n't
Let Nobody Turn You Around '' -- 2 : 27 12 '' maxi ( 1990 ) `` The Joker '' ( LP version ) -- 4 : 22 `` Do n't Let Nobody Turn You Around '' -- 2 : 27 `` Shu Ba
Da Du Ma Ma Ma '' -- 5 : 39 CD maxi ( 1990 ) `` The Joker '' ( single version ) -- 3 : 34 `` Do n't Let Nobody Turn You Around '' -- 2 : 27 `` Shu Ba Da Du Ma
Ma Ma Ma '' -- 3 : 33 `` Living in the U.S.A. ''
</item 2>

<item 3>
The Joker (Steve Miller Band song)
Steve Miller recorded an unreleased demo of this song 's music with the lyrics of his other song , Take the Money and Run . The 2012 song Diese Tage with German
and English lyrics by KRIS feat Dante Thomas uses the chorus of `` The Joker '' . Country rapper Colt Ford sampled this song for his song `` Gangsta of Love ''
. `` Wow Wow '' by Neil Cicierega from the album `` Mouth Moods '' mixes `` The Joker '' with Will Smith 's `` Wild Wild West '' , `` All Star '' by Smash Mouth
and `` Cowboy '' by Kid Rock . In popular culture ( edit ) In 1996 , `` The Joker '' inspired a comedy film , The Pompatus of Love . References and footnotes (
edit ) ^ Jump up to : `` Steve Miller Band > Charts & Awards > Billboard Singles '' . allmusic . Retrieved October 5 , 2009 . ^ Jump up to : `` Chart Stats -
The Steve Miller Band - The Joker '' . Retrieved October 5 , 2009 . Jump up ^ `` irishcharts.ie search results '' .
</item 3>

<item 4>
The Joker (Steve Miller Band song)
The Joker ( Steve Miller Band song ) - wikipedia The Joker ( Steve Miller Band song ) Jump to : navigation , search `` The Joker ( song ) '' redirects here .
For other songs with the same title , see Joker ( disambiguation ) . `` The Joker '' Cover of the 1973 single Single by Steve Miller Band from the album The
Joker B - side `` ' see track listings ' '' Released October 1973 Format 7 '' Single , 12 '' single , Cassette single ( Original Releases ) , CD single and
Digital download ( Post Releases ) . Genre Rock , blues rock , southern rock Length 4 : 26 ( Album version ) ; 3 : 35 ( 7 '' version / Radio edit ) and 6 : 20 (
Extended version ) Label Capitol Records ; EMI - Odeon ; London Records , Parlophone Records , EMI Bovema , Odeon Records .
</item 4>

<item 5>
The Joker (Steve Miller Band song)
Luke Bryan often performs the song near the end of his concerts . Sampling ( edit ) `` The Joker '' was sampled in the song `` Summer and Lightning '' by
Electric Light Orchestra from their 1977 album , Out of the Blue . `` The Joker '' serves as the outline of the song `` So Cal Loco ( Party Like a Rockstar ) ''
by Sprung Monkey which appeared in the film Dude , Where 's My Car ? . It was also sampled on the original Def American Records release of the Geto Boys ' song
`` Gangster of Love '' from their 1989 album , Grip It ! On That Other Level . The sample , presumably unauthorized , was replaced with Lynyrd Skynyrd 's ``
Sweet Home Alabama '' on later pressings of the album . The song `` Hudson Dragonfly '' by Jackie - O Motherfucker , from the album , America Mystica ( 2006 ) ,
contains a sample of `` The Joker '' . The song `` Angel '' by Shaggy , from the album Hot Shot ( 2000 ) , samples the bass line of `` The Joker '' .
</item 5>

<item 6>
The Joker (Steve Miller Band song)
1990 ( two weeks ) Succeeded by `` Show Me Heaven '' by Maria McKee Preceded by `` Trippin ' '' by Push Push New Zealand RIANZ number - one single May 10 , 1991
( one week ) Succeeded by `` Trippin ' '' by Push Push Cover versions ( edit ) `` The Joker '' Single by k.d. lang from the album Drag Released Length 4 : 44
Label Warner Bros . Songwriter ( s ) Eddie Curtis , Ahmet Ertegün , Steve Miller Producer ( s ) k.d. lang , Craig Street k.d.
</item 6>

<item 7>
The Joker (Steve Miller Band song)
( live ) ( 1983 ) String Module Error : Match not found1983 `` Take the Money and Run '' ( live ) / `` The Joker '' ( live ) ( 1983 ) String Module Error :
Match not found1983 `` Shangri - La '' ( 1984 ) Shangri - La1984 `` The Joker '' ( 1990 , reissue ) Cover of the 1990 reissue Steve Miller Band singles
chronology `` Ya Ya '' ( 1988 ) Ya Ya1988 `` The Joker '' ( 1990 ) The Joker1990 `` Wide River '' ( 1993 ) Wide River1993 `` The Joker '' is a song by the Steve
Miller Band from their 1973 album The Joker . It is one of two Steve Miller Band songs that feature the nonce word `` pompatus '' . The song topped the US
Billboard Hot 100 in early 1974 .
</item 7>

<item 8>
The Joker (Steve Miller Band song)
The song topped the US Billboard Hot 100 in early 1974 . More than 16 years later , in September 1990 , it reached number one in the UK Singles Chart for two
weeks after being used in `` Great Deal '' , a Hugh Johnson - directed television advertisement for Levi 's , thus holding the record for the longest gap
between transatlantic chart - toppers . This reissue of `` The Joker '' also topped the Irish Singles Chart , the New Zealand RIANZ Singles Chart , the Dutch
Nationale Top 100 and the Dutch Top 40 . The first line of the lyrics is a reference to the song `` Space Cowboy '' from Miller 's Brave New World album .
Following lines refer to two other songs : `` Gangster of Love '' from Sailor and `` Enter Maurice '' from Recall the Beginning ... A Journey from Eden .
</item 8>

<item 9>
The Joker (Steve Miller Band song)
Retrieved October 6 , 2009 . Jump up ^ `` The Irish Charts - search result '' . Retrieved May 16 , 2016 . Jump up ^ `` norwegiancharts.com - Steve Miller Band -
The Joker '' . Retrieved October 6 , 2009 . Jump up ^ `` swedishcharts.com - Steve Miller Band - The Joker '' . Retrieved October 6 , 2009 . Jump up ^ `` Steve
Miller Band - The Joker - hitparade.ch '' ( in German ) . Retrieved October 6 , 2009 . Jump up ^ `` End of Year Charts 1991 '' . Recorded Music NZ . Retrieved
December 3 , 2017 . Jump up ^ `` Chart Stats - Fatboy Slim - The Joker '' . Retrieved October 13 , 2009 . Jump up ^ `` Smashing Pumpkins dot com '' . Retrieved
September 28 , 2009 . Jump up ^ `` The Howard Stern Radio Show '' . New York . 17 Apr 2013 . Sirius XM Satellite Radio . Howard 100 .
</item 9>

<item 10>
The Joker (Steve Miller Band song)
Songwriter ( s ) Eddie Curtis , Ahmet Ertegün , Steve Miller Producer ( s ) Steve Miller Steve Miller Band singles chronology `` Fandango '' ( 1972 ) `` The
Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash '' ( 1974 ) `` Fandango '' ( 1972 ) `` The Joker '' ( 1973 ) `` Your Cash Ai n't Nothin ' But Trash ''
( 1974 ) `` Take the Money and Run '' / `` The Joker '' ( 1983 , double A-side , live ) Cover of the 1983 live single Steve Miller Band singles chronology ``
Living in the U.S.A. ''
</item 10>
=== END ITEM id=0e72630704aac3a4 ===
