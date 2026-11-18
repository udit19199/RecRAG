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

This file has 6 items.

=== ITEM id=dc4f39cd4f4c5891 ===
Gold sentences and facts:
Sentence 0: The right-most label conveys the top - level domain ; for example , the domain name www.example.com belongs to the top - level domain com . The hierarchy of domains descends from the right to the left label in the name ; each label to the left specifies a subdivision , or subdomain of the domain to the right . For example : the label example specifies a node example.com as a subdomain of the com domain , and www is a label to create www.example.com , a subdomain of example.com . Each label may contain from 1 to 63 octets . The empty label is reserved for the root node and when fully qualified is expressed as the empty label terminated by a dot . The full domain name may not exceed a total length of 253 ASCII characters in its textual representation . Thus , when using a single character per label , the limit is 127 levels : 127 characters plus 126 dots have a total length of 253 . In practice , some domain registries may have shorter limits . A hostname is a domain name that has at least one associated IP address . For example , the domain names www.example.com and example.com are also hostnames , whereas the com domain is not . However , other top - level domains , particularly country code top - level domains , may indeed have an IP address , and if so , they are also hostnames . Hostnames impose restrictions on the characters allowed in the corresponding domain name . A valid hostname is also a valid domain name , but a valid domain name may not necessarily be valid as a hostname .
  Fact 0.0: The right-most label of a domain name conveys the top-level domain.
  Fact 0.1: In www.example.com, the top-level domain is com.
  Fact 0.2: The hierarchy of domains descends from the right label to the left label in the name.
  Fact 0.3: Each label to the left specifies a subdivision, or subdomain, of the domain to its right.
  Fact 0.4: The label example specifies the node example.com as a subdomain of the com domain.
  Fact 0.5: The label www creates www.example.com, a subdomain of example.com.
  Fact 0.6: Each label may contain from 1 to 63 octets.
  Fact 0.7: The empty label is reserved for the root node.
  Fact 0.8: When fully qualified, the root node is expressed as the empty label terminated by a dot.
  Fact 0.9: The full domain name may not exceed a total length of 253 ASCII characters in its textual representation.
  Fact 0.10: When using a single character per label, the limit is 127 levels.
  Fact 0.11: 127 characters plus 126 dots have a total length of 253.
  Fact 0.12: In practice, some domain registries may have shorter limits.
  Fact 0.13: A hostname is a domain name that has at least one associated IP address.
  Fact 0.14: The domain names www.example.com and example.com are also hostnames.
  Fact 0.15: The com domain is not a hostname.
  Fact 0.16: Other top-level domains, particularly country code top-level domains, may have an IP address.
  Fact 0.17: If another top-level domain has an IP address, it is also a hostname.
  Fact 0.18: Hostnames impose restrictions on the characters allowed in the corresponding domain name.
  Fact 0.19: A valid hostname is also a valid domain name.
  Fact 0.20: A valid domain name may not necessarily be valid as a hostname.

Retrieved items:
<item 1>
Domain name
Host names appear as a component in Uniform Resource Locators ( URLs ) for Internet resources such as web sites ( e.g. , en.wikipedia.org ) . Domain names are
also used as simple identification labels to indicate ownership or control of a resource . Such examples are the realm identifiers used in the Session
Initiation Protocol ( SIP ) , the Domain Keys used to verify DNS domains in e-mail systems , and in many other Uniform Resource Identifiers ( URIs ) . An
important function of domain names is to provide easily recognizable and memorizable names to numerically addressed Internet resources . This abstraction allows
any resource to be moved to a different physical location in the address topology of the network , globally or locally in an intranet . Such a move usually
requires changing the IP address of a resource and the corresponding translation of this IP address to and from its domain name . Domain names are used to
establish a unique identity . Organizations can choose a domain name that corresponds to their name , helping Internet users to reach them easily . A generic
domain is a name that defines a general category , rather than a specific or personal instance , for example , the name of an industry , rather than a company
name .
</item 1>

<item 2>
Domain name
Each label is separated by a full stop ( dot ) . ' sos ' is said to be a sub-domain of ' state.oh.us ' , and ' state ' a sub-domain of ' oh.us ' , etc . In
general , subdomains are domains subordinate to their parent domain . An example of very deep levels of subdomain ordering are the IPv6 reverse resolution DNS
zones , e.g. , 1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa , which is the reverse DNS resolution domain name for the IP address of
a loopback interface , or the localhost name . Second - level ( or lower - level , depending on the established parent hierarchy ) domain names are often
created based on the name of a company ( e.g. , bbc.co.uk ) , product or service ( e.g. hotmail.com ) . Below these levels , the next domain name component has
been used to designate a particular host server .
</item 2>

<item 3>
Domain name
Labels are always separated by the full stop ( period ) character in the textual name representation . Business models Domain names are often seen in analogy to
real estate in that domain names are foundations on which a website can be built , and the highest quality domain names , like sought - after real estate , tend
to carry significant value , usually due to their online brand - building potential , use in advertising , search engine optimization , and many other criteria
. A few companies have offered low - cost , below - cost or even free domain registration with a variety of models adopted to recoup the costs to the provider .
These usually require that domains be hosted on their website within a framework or portal that includes advertising wrapped around the domain holder 's content
, revenue from which allows the provider to recoup the costs . Domain registrations were free of charge when the DNS was new . A domain holder may provide
infinite number of subdomains in their domain . For example , the owner of example.org could provide subdomains such as foo.example.org and foo.bar.example.org
to interested parties . Many desirable domain names are already assigned and users must search for other acceptable names , using Web - based search features ,
or WHOIS and dig operating system tools .
</item 3>

<item 4>
Domain name
The first - level set of domain names are the top - level domains ( TLDs ) , including the generic top - level domains ( gTLDs ) , such as the prominent domains
com , info , net , edu , and org , and the country code top - level domains ( ccTLDs ) . Below these top - level domains in the DNS hierarchy are the second -
level and third - level domain names that are typically open for reservation by end - users who wish to connect local area networks to the Internet , create
other publicly accessible Internet resources or run web sites . The registration of these domain names is usually administered by domain name registrars who
sell their services to the public . A fully qualified domain name ( FQDN ) is a domain name that is completely specified with all labels in the hierarchy of the
DNS , having no parts omitted . Labels in the Domain Name System are case - insensitive , and may therefore be written in any desired capitalization method ,
but most commonly domain names are written in lowercase in technical contexts .
</item 4>

<item 5>
Domain name
Domain name - wikipedia Domain name Jump to : navigation , search This article is about domain names in the Internet . For other uses , see Domain . The
hierarchy of labels in a fully qualified domain name . A domain name is an identification string that defines a realm of administrative autonomy , authority or
control within the Internet . Domain names are formed by the rules and procedures of the Domain Name System ( DNS ) . Any name registered in the DNS is a domain
name . Domain names are used in various networking contexts and for application - specific naming and addressing purposes . In general , a domain name
represents an Internet Protocol ( IP ) resource , such as a personal computer used to access the Internet , a server computer hosting a web site , or the web
site itself or any other service communicated via the Internet . In 2017 , 330.6 million domain names had been registered . Domain names are organized in
subordinate levels ( subdomains ) of the DNS root domain , which is nameless .
</item 5>

<item 6>
Domain name
For special purposes , such as network testing , documentation , and other applications , IANA also reserves a set of special - use domain names . This list
contains domain names such as example , local , localhost , and test . Other top - level domain names containing trade marks are registered for corporate use .
Cases include brands such as BMW , Google , and Canon . Second - level and lower level domains Below the top - level domains in the domain name hierarchy are
the second - level domain ( SLD ) names . These are the names directly to the left of . com , . net , and the other top - level domains . As an example , in the
domain example.co.uk , co is the second - level domain . Next are third - level domains , which are written immediately to the left of a second - level domain .
There can be fourth - and fifth - level domains , and so on , with virtually no limitation . An example of an operational domain name with four levels of domain
labels is sos.state.oh.us . Each label is separated by a full stop ( dot ) . '
</item 6>

<item 7>
Domain name
Domain name space Today , the Internet Corporation for Assigned Names and Numbers ( ICANN ) manages the top - level development and architecture of the Internet
domain name space . It authorizes domain name registrars , through which domain names may be registered and reassigned . The hierarchical domain name system ,
organized into zones , each served by domain name servers . The domain name space consists of a tree of domain names . Each node in the tree holds information
associated with the domain name . The tree sub-divides into zones beginning at the DNS root zone . Domain name syntax A domain name consists of one or more
parts , technically called labels , that are conventionally concatenated , and delimited by dots , such as example.com . The right-most label conveys the top -
level domain ; for example , the domain name www.example.com belongs to the top - level domain com . The hierarchy of domains descends from the right to the
left label in the name ; each label to the left specifies a subdivision , or subdomain of the domain to the right .
</item 7>

<item 8>
Domain name
Hostnames impose restrictions on the characters allowed in the corresponding domain name . A valid hostname is also a valid domain name , but a valid domain
name may not necessarily be valid as a hostname . Top - level domains The top - level domains ( TLDs ) such as com , net and org are the highest level of domain
names of the Internet . Top - level domains form the DNS root zone of the hierarchical Domain Name System . Every domain name ends with a top - level domain
label . When the Domain Name System was devised in the 1980s , the domain name space was divided into two main groups of domains . The country code top - level
domains ( ccTLD ) were primarily based on the two - character territory codes of ISO - 3166 country abbreviations . In addition , a group of seven generic top -
level domains ( gTLD ) was implemented which represented a set of categories of names and multi-organizations . These were the domains gov , edu , com , mil ,
org , net , and int . During the growth of the Internet , it became desirable to create additional generic top - level domains .
</item 8>

<item 9>
Domain name
For example : the label example specifies a node example.com as a subdomain of the com domain , and www is a label to create www.example.com , a subdomain of
example.com . Each label may contain from 1 to 63 octets . The empty label is reserved for the root node and when fully qualified is expressed as the empty
label terminated by a dot . The full domain name may not exceed a total length of 253 ASCII characters in its textual representation . Thus , when using a
single character per label , the limit is 127 levels : 127 characters plus 126 dots have a total length of 253 . In practice , some domain registries may have
shorter limits . A hostname is a domain name that has at least one associated IP address . For example , the domain names www.example.com and example.com are
also hostnames , whereas the com domain is not . However , other top - level domains , particularly country code top - level domains , may indeed have an IP
address , and if so , they are also hostnames . Hostnames impose restrictions on the characters allowed in the corresponding domain name .
</item 9>

<item 10>
Domain name
Below these levels , the next domain name component has been used to designate a particular host server . Therefore , ftp.example.com might be an FTP server ,
www.example.com would be a World Wide Web server , and mail.example.com could be an email server , each intended to perform only the implied function . Modern
technology allows multiple physical servers with either different ( cf . load balancing ) or even identical addresses ( cf . anycast ) to serve a single
hostname or domain name , or multiple domain names to be served by a single computer . The latter is very popular in Web hosting service centers , where service
providers host the websites of many organizations on just a few servers . The hierarchical DNS labels or components of domain names are separated in a fully
qualified name by the full stop ( dot , . ) . Internationalized domain names Main article : Internationalized domain name The character set allowed in the
Domain Name System is based on ASCII and does not allow the representation of names and words of many languages in their native scripts or alphabets . ICANN
approved the Internationalized domain name ( IDNA ) system , which maps Unicode strings used in application user interfaces into the valid DNS character set by
an encoding called Punycode .
</item 10>
=== END ITEM id=dc4f39cd4f4c5891 ===

=== ITEM id=4f918ccb1ad0167a ===
Gold sentences and facts:
Sentence 0: Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 - 13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
  Fact 0.0: The table lists 17 episodes with first airing on September 13, 1969 and last airing on January 17, 1970.
  Fact 0.1: The table lists 8 episodes with first airing on September 12, 1970 and last airing on October 31, 1970.
  Fact 0.2: The table lists 16 episodes with first airing on September 9, 1978 and last airing on December 23, 1978.

Retrieved items:
<item 1>
List of Scooby-Doo, Where Are You! episodes
The only way to claim the inheritance is for Scooby , and the other four heirs to stay the night in the Colonel 's mansion -- which they quickly learn is
haunted by two phantom shadows . Note : In `` Scoobynatural , '' a season 13 episode of The CW horror fantasy show Supernatural , protagonists Sam and Dean
Winchester along with the angel Castiel get sucked into this episode of Scooby - Doo by an actual ghost where they team up with the Scooby Gang to solve the
mystery . 17 17 `` That 's Snow Ghost '' Snow Ghost / Mr. Greenway January 17 , 1970 ( 1970 - 01 - 17 ) While on a skiing vacation at Wolf 's End Lodge , the
kids are scared by the ghost of a Yeti , called the Snow Ghost , who is believed to turn anyone he captures into ghosts . In search of clues , the gang
eventually meet a Tibetan hermit who thinks he is responsible for summoning the monster . Season 2 ( 1970 ) ( edit ) No . overall No.
</item 1>

<item 2>
List of Scooby-Doo, Where Are You! episodes
11 11 `` A Gaggle of Galloping Ghosts '' Count Dracula , Werewolf , Frankenstein 's Monster and the Gypsy / Big Bob Oakley , alias `` The Actor '' November 22 ,
1969 ( 1969 - 11 - 22 ) While on their way to Franken Castle , the gang stop to get their fortunes told by a gypsy . She warns them `` they will meet their doom
'' if they go to the castle . The gang goes to the castle anyway , where they are confronted by Count Dracula , Frankenstein 's monster and a Werewolf . 12 12
`` Scooby - Doo and a Mummy , Too '' Mummy of Ankha / Dr. Najib November 29 , 1969 ( 1969 - 11 - 29 ) While visiting a museum , the gang learn about the legend
of Ankha , a 3000 - year - old mummy . Legend had it that the mummy will come back alive and turn people to stone if he is removed from his tomb .
</item 2>

<item 3>
List of Scooby-Doo, Where Are You! episodes
episodes ( 1969 -- 1970 ) The New Scooby - Doo Movies episodes ( 1972 -- 73 ) The Scooby - Doo Show episodes ( 1976 -- 78 ) Scooby - Doo and Scrappy - Doo (
1979 -- 1980 ) Scooby - Doo and Scrappy - Doo ( 1980 -- 82 ) The New Scooby and Scrappy - Doo Show / The New Scooby - Doo Mysteries ( 1983 -- 84 ) The 13 Ghosts
of Scooby - Doo ( 1985 ) A Pup Named Scooby - Doo episodes ( 1988 -- 1991 ) What 's New , Scooby - Doo ? episodes ( 2002 -- 06 ) Shaggy & Scooby - Doo Get a
Clue ! episodes ( 2006 -- 08 ) Scooby - Doo ! Mystery Incorporated episodes ( 2010 -- 13 ) Be Cool , Scooby - Doo !
</item 3>

<item 4>
List of Scooby-Doo, Where Are You! episodes
9 9 `` The Backstage Rage '' Phantom Puppeteer / Mr. Pietro November 8 , 1969 ( 1969 - 11 - 08 ) While getting a pizza , Shaggy and Scooby - Doo come across a
violin case filled with counterfeit money . While Scooby is distracted by a dog puppet , the case of money is stolen away . Shaggy , Scooby , and the rest of
the gang find a string puppet controller at the scene and follow it to the local puppet theater , where they uncover a counterfeiting operation , controlled by
a spooky puppet master . 10 10 `` Bedlam in the Big Top '' Ghost Clown / Harry the Hypnotist November 15 , 1969 ( 1969 - 11 - 15 ) Mystery , Inc. meets a
strongman and a midget , both on the run from a circus allegedly haunted by a ghost clown . The gang goes to the circus grounds to investigate , but several of
them fall victim to the Ghost Clown 's powers of hypnotism .
</item 4>

<item 5>
List of Scooby-Doo, Where Are You! episodes
is a Saturday morning cartoon produced by Hanna - Barbera . Contents ( hide ) 1 Series overview 2 Episodes 2.1 Season 1 ( 1969 -- 70 ) 2.2 Season 2 ( 1970 ) 2.3
Season 3 ( 1978 ) ( Revival ) 3 References Series overview ( edit ) Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 -
13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
Episodes ( edit ) Season 1 ( 1969 -- 70 ) ( edit ) No . overall No. in season Title Villain ( s ) Original air date `` What a Night for a Knight '' Black Knight
/ Mr.
</item 5>

<item 6>
List of Scooby-Doo, Where Are You! episodes
banner , as the revival series was cancelled . 35 10 `` The Creepy Creature of Vulture 's Claw '' Mantis Creature / Professor Greer November 11 , 1978 ( 1978 -
11 - 11 ) The gang discovers a humanoid mantis creature roaming around Vulture 's Claw Botanical Gardens . 36 11 `` The Diabolical Disc Demon '' The Phantom /
Ace Decade November 18 , 1978 ( 1978 - 11 - 18 ) The kids discovers the Diabolical Disc Demon . He wants a half - finished song of a missing writer and will
stop at nothing to get it . 37 12 `` Scooby 's Chinese Fortune Kooky Caper '' Moon Monster / Uncle Chin November 25 , 1978 ( 1978 - 11 - 25 ) The gang runs into
The Moon Monster . There are rumors that if you run into his shadow , you will be turned into stone .
</item 6>

<item 7>
List of Scooby-Doo, Where Are You! episodes
( 2015 -- 18 ) Package shows and programming blocks The Scooby - Doo / Dynomutt Hour ( 1976 -- 77 ) Scooby 's All - Star Laff - A-Lympics / Scooby 's All -
Stars ( 1977 -- 78 ) The Richie Rich / Scooby - Doo Show ( 1980 -- 81 ) The Scooby & Scrappy - Doo / Puppy Hour ( 1982 ) Scary Scooby Funnies ( 1984 -- 85 )
Scooby 's Mystery Funhouse ( 1985 -- 86 ) Television films and specials Hanna - Barbera 's All - Star Comedy Ice Revue ( 1978 ) Scooby Goes Hollywood ( 1979 )
Scooby - Doo Meets the Boo Brothers ( 1987 ) Scooby - Doo and the Ghoul School ( 1988 ) Scooby - Doo ! and the Reluctant Werewolf ( 1988 ) Hanna - Barbera 's
50th : A Yabba Dabba Doo Celebration ( 1989 ) Scooby - Doo !
</item 7>

<item 8>
List of Scooby-Doo, Where Are You! episodes
Wickles September 13 , 1969 ( 1969 - 09 - 13 ) While walking home from the movies , Shaggy and Scooby - Doo discover a black suit of armor in a pickup truck ,
and the gang deliver it to the local museum . The kids later break into the museum to search for clues , after learning that the archaeologist who was
transporting the suit is missing , and they find that the knight 's armor has come to life . `` Hassle in the Castle '' Phantom of Vasquez / Bluestone the Great
September 20 , 1969 ( 1969 - 09 - 20 ) The gang goes out on a boating expedition , but gets lost in a fog and crashes their boat on the mysterious Haunted Isle
. They decide to venture to the abandoned Vázquez Castle on the island for help , but instead encounter a transparent phantom running loose , who threatens them
to leave the island and never return .
</item 8>

<item 9>
List of Scooby-Doo, Where Are You! episodes
5 5 `` Decoy for a Dognapper '' Indian Witch Doctor and Ghost of Geronimo / Buck Masters October 11 , 1969 ( 1969 - 10 - 11 ) When a series of local dognappings
alarms prize - winning dog trainer Buck Masters , the gang offers to have Scooby - Doo pose as a decoy to catch the dognappers . However , their plan works too
well as Scooby is himself dognapped , leading the gang to discover that the dognappers appear to be under the command of a ghostly Indian witch doctor . 6 6 ``
What the Hex Going On ? '' Ghost of Elias Kingston / Stuart Weatherby October 18 , 1969 ( 1969 - 10 - 18 ) The kids pay a visit to their friend Sharon Weatherby
, whose Uncle Stuart has been cursed by the ghost of the family 's neighbor , Elias Kingston . When Uncle Stuart disappears altogether , Mystery , Inc. ventures
into the abandoned and allegedly haunted Kingston Mansion next door , in search of him and the ghost .
</item 9>

<item 10>
List of Scooby-Doo, Where Are You! episodes
21 `` Scooby 's Night with a Frozen Fright '' Caveman / Professor Wayne Seven Days a Week October 3 , 1970 ( 1970 - 10 - 03 ) While out fishing , Shaggy and
Scooby - Doo hook a two - million - year - old caveman frozen in a block of ice , which the gang learns was lost at sea during a violent storm . They take the
caveman to the Oceanland aquarium , but when someone melts the ice block and the caveman goes on the rampage , the gang decides to stick around and investigate
. 22 5 `` Haunted House Hang - Up '' Headless Spectre / Penrod Stillwall , Phantom / Asa Shanks Love the World October 10 , 1970 ( 1970 - 10 - 10 ) The Mystery
Machine overheats in front of an old spooky mansion , and the gang goes to look for help , but finds out that the house is being haunted by the Headless Spectre
, a terrifying phantom with no head . The mystery gets even stranger when they begin to hear knocking sounds and clues leading to a treasure .
</item 10>
=== END ITEM id=4f918ccb1ad0167a ===

=== ITEM id=18dd6c546b7ba381 ===
Gold sentences and facts:
Sentence 0: Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi. tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16 urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China .
  Fact 0.0: Beijing is the capital of the People's Republic of China.
  Fact 0.1: Beijing was formerly romanized as Peking.
  Fact 0.2: Beijing is the world's second most populous city proper.
  Fact 0.3: Beijing is the most populous capital city.
  Fact 0.4: Beijing is located in northern China.
  Fact 0.5: Beijing is governed as a direct-controlled municipality under the national government.
  Fact 0.6: Beijing has 16 urban, suburban, and rural districts.
  Fact 0.7: Beijing Municipality is surrounded by Hebei Province, except for the neighboring Tianjin Municipality to the southeast.
  Fact 0.8: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the Jingjinji metropolitan region.
  Fact 0.9: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the national capital region of China.
Sentence 1: Over the past 3,000 years , the city of Beijing has had numerous other names . The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University , still use the former romanization .
  Fact 1.0: Over the past 3,000 years, the city of Beijing has had numerous other names.
  Fact 1.1: The name Beijing means 'Northern Capital'.
  Fact 1.2: The name Beijing is made of the Chinese characters 北 (north) and 京 (capital).
  Fact 1.3: The name Beijing was applied to the city in 1403 during the Ming dynasty.
  Fact 1.4: The name Beijing was applied to distinguish the city from Nanjing, the 'Southern Capital'.
  Fact 1.5: The English spelling Beijing is based on the pinyin romanization of the two characters as pronounced in Standard Mandarin.
  Fact 1.6: The older English spelling Peking is the postal romanization of the same two characters as pronounced in Chinese dialects spoken in the southern port towns.
  Fact 1.7: The southern port towns were the first visited by European traders and missionaries.
  Fact 1.8: The Chinese dialects spoken in the southern port towns preserve the Middle Chinese pronunciation of 京 as kjaeng.
  Fact 1.9: That pronunciation of 京 as kjaeng comes before a phonetic shift in the northern dialects to the modern pronunciation.
  Fact 1.10: Peking is no longer the common name for the city.
  Fact 1.11: Some of the city's older locations and facilities still use the former romanization Peking.
  Fact 1.12: Beijing Capital International Airport, with IATA code PEK, still uses the former romanization.
  Fact 1.13: Peking University still uses the former romanization.

Retrieved items:
<item 1>
Beijing
1928 -- 1937 ; 1945 -- 1949 ) Literal meaning `` Northern Capital '' ( show ) Transcriptions Standard Mandarin Hanyu Pinyin Běijīng Bopomofo ㄅ ㄟ ˇ ㄐ ㄧ ㄥ Gwoyeu
Romatzyh Beeijing Wade -- Giles Pei - ching IPA ( pèi. tɕíŋ ) Wu Suzhounese Poh - cin Hakka Romanization Bet - gin Yue : Cantonese Yale Romanization Bākgìng or
Bākgīng IPA ( pɐ́k̚. kêŋ ) or ( pɐ́k̚. kéŋ ) Jyutping Bak1ging1 Southern Min Hokkien POJ Pak - kian Tâi - lô Pak - kiann Eastern Min Fuzhou BUC Báe̤k - gĭng
Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi.
</item 1>

<item 2>
Beijing
Retrieved 22 July 2009 . External links Find more aboutBeijingat Wikipedia 's sister projects Definitions from Wiktionary Media from Wikimedia Commons News from
Wikinews Quotations from Wikiquote Texts from Wikisource Textbooks from Wikibooks Travel guide from Wikivoyage Learning resources from Wikiversity Economic
profile for Beijing at HKTDC Visit Beijing Facebook Page Photograph of The approach to Peking -- outside the walls taken in 1890 by Sir Henry Norman Preceded by
Lin'an ( Song dynasty ) Capital of China ( as Dadu of Yuan ) 1264 -- 1368 Succeeded by Nanjing ( Ming dynasty ) Preceded by Nanjing ( Ming dynasty ) Capital of
China 1420 -- 1928 Succeeded by Nanjing ( ROC ) Preceded by Nanjing ( ROC ) Capital of the People 's Republic of China 1949 -- present Succeeded by present
capital Places adjacent to Beijing Zhangjiakou , Hebei Chengde , Hebei Chengde , Hebei Zhangjiakou , Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang ,
</item 2>

<item 3>
Beijing
And under the Hongwu Emperor 's feudal policies it was given to Zhu Di , one of his sons , who was created `` Prince of Yan '' . The early death of Zhu
Yuanzhang 's heir led to a succession struggle on his death , one that ended with the victory of Zhu Di and the declaration of the new Yongle era . Since his
harsh treatment of the Ming capital Yingtian ( modern Nanjing ) alienated many there , he established his fief as a new co-capital . The city of Beiping became
Beijing ( northern capital ) or Shuntian in 1403 . The construction of the new imperial residence , the Forbidden City , took from 1406 to 1420 ; this period
was also responsible for several other of the modern city 's major attractions , such as the Temple of Heaven and Tian'anmen . On 28 October 1420 , the city was
officially designated the capital of the Ming dynasty in the same year that the Forbidden City was completed . Beijing became the empire 's primary capital and
Yingtian , also called Nanjing ( southern capital ) , became the co-capital .
</item 3>

<item 4>
Beijing
The official Latin alphabet abbreviation for Beijing is `` BJ '' . History Main article : History of Beijing Early History The earliest traces of human
habitation in the Beijing municipality were found in the caves of Dragon Bone Hill near the village of Zhoukoudian in Fangshan District , where Peking Man lived
. Homo erectus fossils from the caves date to 230,000 to 250,000 years ago . Paleolithic Homo sapiens also lived there more recently , about 27,000 years ago .
Archaeologists have found neolithic settlements throughout the municipality , including in Wangfujing , located in downtown Beijing . The first walled city in
Beijing was Jicheng , the capital city of the state of Ji and was built in 1045 BC . Within modern Beijing , Jicheng was located around the present Guang'anmen
area in the south of Xicheng District . This settlement was later conquered by the state of Yan and made its capital . Early Imperial China The Tianning Pagoda
, built around 1120 during the Liao dynasty . After the First Emperor unified China , Jicheng became a prefectural capital for the region .
</item 4>

<item 5>
Beijing
Two generations later , Kublai Khan ordered the construction of Dadu ( or Daidu to the Mongols , commonly known as Khanbaliq ) , a new capital for his Yuan
dynasty to the northeast of the Zhongdu ruins . The construction took from 1264 to 1293 , but greatly enhanced the status of a city on the northern fringe of
China proper . The city was centered on the Drum Tower slightly to the north of modern Beijing and stretched from the present - day Chang'an Avenue to the
northern part of Line 10 subway . Remnants of the Yuan rammed earth wall still stand and are known as the Tucheng . Ming dynasty One of the corner towers of the
Forbidden City , which was built by Emperor Yongle in Ming dynasty . In 1368 , soon after declaring the new Hongwu era of the Ming dynasty , the rebel leader
Zhu Yuanzhang sent an army to Khanbaliq and conquered it . Since the Yuan continued to occupy Shangdu and Mongolia , however , Dadu was renamed to Beiping as
used to supply the military garrisons in the area .
</item 5>

<item 6>
Beijing
The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the
Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two
characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are
pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese
pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for
the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University ,
still use the former romanization . The single Chinese character abbreviation for Beijing is 京 , which appears on automobile license plates in the city . The
official Latin alphabet abbreviation for Beijing is `` BJ '' .
</item 6>

<item 7>
Beijing
On 28 June the same year , Beijing 's name was returned to Beiping ( written at the time as `` Peiping '' ) . On July 7 , 1937 , the 29th Army and the Japanese
army in China exchanged fire at the Marco Polo Bridge near the Wanping Fortress southwest of the city . The Marco Polo Bridge Incident triggered the Second Sino
- Japanese War , World War II as it is known in China . During the war , Beiping fell to Japan on 29 July 1937 and was made the seat of the Provisional
Government of the Republic of China , a puppet state that ruled the ethnic - Chinese portions of Japanese - occupied northern China . This government was later
merged into the larger Wang Jingwei government based in Nanjing . People 's Republic of China Mao Zedong proclaiming the establishment of the People 's Republic
of China in 1949 In the final phases of the Chinese Civil War , the People 's Liberation Army seized control of the city peacefully on 31 January 1949 in the
course of the Pingjin Campaign .
</item 7>

<item 8>
Beijing
Beijing - wikipedia Beijing Jump to : navigation , search `` Peking '' redirects here . For other uses , see Beijing ( disambiguation ) and Peking (
disambiguation ) . Beijing 北京 市 Municipality Beijing Municipality Clockwise from top : Beijing CBD skyline , Tiananmen , Temple of Heaven , National Center for
the Performing Arts , and Beijing National Stadium Location of Beijing Municipality within China Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383
° E ﻿ / 39.917 ; 116.383 Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383 ° E ﻿ / 39.917 ; 116.383 Country China Established date 1045 BCE ( Zhou
Dynasty ) Divisions - County - level - Township - level 16 districts 289 towns and villages Government Type Municipality Party Secretary Cai Qi Mayor Chen
Jining Congress Chairman Li Wei ( zh ) Conference Chairman Ji Lin Area Municipality 16,411 km ( 6,336 sq mi ) Land 16,
</item 8>

<item 9>
Beijing
In 938 , after the fall of the Tang , the Later Jin ceded the entire northern frontier to the Khitan Liao dynasty , which treated the city as Nanjing , or the
`` Southern Capital '' , one of four secondary capitals to complement its `` Supreme Capital '' , Shangjing ( modern Baarin Left Banner in Inner Mongolia ) .
Some of the oldest surviving structures in Beijing date to the Liao period , including the Tianning Pagoda . Miaoying Temple and its White Stupa in Dadu which
was built by Kublai Khan The Liao fell to the Jurchen Jin dynasty in 1122 , which gave the city to the Song dynasty and then retook it in 1125 during its
conquest of northern China . In 1153 , the Jurchen Jin made Beijing their `` Central Capital '' , or Zhongdu . The city was besieged by Genghis Khan 's invading
Mongolian army in 1213 and razed to the ground two years later .
</item 9>

<item 10>
Beijing
After the First Emperor unified China , Jicheng became a prefectural capital for the region . During the Three Kingdoms period , it was held by Gongsun Zan and
Yuan Shao before falling to Cao Cao 's Wei Kingdom . The AD 3rd - century Western Jin demoted the town , placing the prefectural seat in neighboring Zhuozhou .
During the Sixteen Kingdoms period when northern China was conquered and divided by the Wu Hu , Jicheng was briefly the capital of the Xianbei Former Yan
Kingdom . After China was reunified during the Sui dynasty , Jicheng , also known as Zhuojun , became the northern terminus of the Grand Canal . Under the Tang
dynasty , Jicheng as Youzhou , served as a military frontier command center . During the An - Shi Rebellion and again amidst the turmoil of the late Tang ,
local military commanders founded their own short - lived Yan dynasties and called the city Yanjing , or the `` Yan Capital . '' Also in the Tang dynasty , the
city 's name Jicheng was replaced by Youzhou or Yanjing .
</item 10>
=== END ITEM id=18dd6c546b7ba381 ===

=== ITEM id=9bd159f1312738c0 ===
Gold sentences and facts:
Sentence 0: Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995
  Fact 0.0: Mumbai has the Marathi name मुंबई.
  Fact 0.1: Mumbai was previously named Bombay.
  Fact 0.2: Mumbai was renamed in 1995.

Retrieved items:
<item 1>
Renaming of cities in India
Realignment of the official Indian English name to an alternative local name ( edit ) Aside from changes to the official English spellings of local names there
have also been renaming proposals to realign the official name , hence the English name with an alternative local name . Ethnically sensitive examples include
the proposals by the Bharatiya Janata Party ( 1990 , 2001 ) to rename Ahmedabad to Karnavati and Allahabad to Prayag . These two proposals are changes from the
Muslim ( or Mughal ) name to a Hindu native name . These can be represented as a change from Urdu language to Hindi language , but since the two languages are
variants of Hindustani the proposal is effectively a cultural and ethno - religious proposal rather than a linguistic one . Adoption of name changes in India
and overseas ( edit ) Official name changes naturally take place quickly if not immediately in official government sources . Adoption may be slower among the
media in India and abroad , and among Indian authors . Important examples ( edit ) Main article : List of renamed places in India States ( edit ) Travancore -
Cochin → Kerala ( 1 November 1956 ) .
</item 1>

<item 2>
Renaming of cities in India
Jump up ^ Calcutta : A Cultural and Literary History - Page 3 Krishna Dutta - 2003 `` nationalist stance , like Bombay , which changed its name to Mumbai , or
Madras , which has become the unrecognisable Chennai , Calcutta has preferred a comparatively minor name change , which frankly is a bit of a multicultural
mishmash . '' Jump up ^ `` David Rumsey : Geographical Searching with MapRank Search ( beta ) '' . Rumsey.mapranksearch.com . Retrieved 2012 - 08 - 15 . Jump up
^ Beam , Christopher ( 2006 - 07 - 12 ) . `` How Bombay became Mumbai . - Slate Magazine '' . Slate.com . Retrieved 2012 - 08 - 15 .
</item 2>

<item 3>
Renaming of cities in India
Madhya Bharat → Madhya Pradesh ( 1 November 1959 ) Madras State → Tamil Nadu ( 14 January 1969 ) Mysore State → Karnataka ( 1 November 1973 ) Uttaranchal →
Uttarakhand ( 1 January 2007 ) Orissa → Odisha ( 4 November 2011 ) Cities ( edit ) Notable city names that were officially changed by legislation after
independence include : Jabalpur ( Hindi : जबलपुर ) , from Jubbulpore , respelled in 1947 Jajmau ( Hindi : जाजमऊ ) , from Jajesmow , respelled in 1948 Kanpur (
Hindi : कानपुर ) , from Cawnpore , respelled in 1948 Vadodara ( Gujarati : વડોદરા ) , from Baroda , respelled in 1974 Thiruvananthapuram ( Malayalam :
തിരുവനന്തപുരം ) , from Trivandrum ,
</item 3>

<item 4>
Renaming of cities in India
The name changes are not without controversy among Indians . In several instances , the name change represents a struggle between a cosmopolitan elite and a
local , regional - language populace over defining the city in ways that go far beyond a simple change of name . '' Jump up ^ Steven I. Wilkinson Votes and
Violence : Electoral Competition and Ethnic Riots in India 2006 Page 23 `` The BJP proposed in 1990 and 2001 that Ahmedabad be renamed `` Karnavati . '' Hindu ,
June 1 1 , 2001 . Similar proposals have been made to rename Allahabad `` Prayag . '' Jump up ^ Cosmopolitanism - Page 73 Carol A. Breckenridge , Sheldon
Pollock , Homi K. Bhabha - 2002 `` In one sense , the decision to officialise the name Mumbai is part of a widespread Indian pattern of replacing names
associated with colonial rule with names associated with local , national , and regional heroes .
</item 4>

<item 5>
Renaming of cities in India
from Trivandrum , respelled in 1991 Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995 Kochi ( Malayalam : കൊച്ചി ) , from Cochin , respelled in 1996
Chennai ( Tamil : சென்னை ) , from Madras , renamed in 1996 Kolkata ( Bengali : কলকাতা ) , from Calcutta , respelled in 2001 Kadapa ( Telugu : కడప ) , from
Cuddapah , respelled in 2005 Puducherry ( Tamil : புதுச்சேரி ) , from Pondicherry , renamed in 2006 Bengaluru ( Kannada : ಬೆಂಗಳೂರು ) , from Bangalore ,
</item 5>

<item 6>
Renaming of cities in India
Renaming of cities in India - Wikipedia Renaming of cities in India Jump to : navigation , search The renaming of cities in India started in 1947 following the
end of the British imperial period . Several changes were politically controversial , and not all proposed changes were implemented . Each had to be approved by
the central government in New Delhi . The renaming of states and territories in India has also taken place , but until recently with actual substantial name
changes in both local language and in English such as the old British state name of Travancore - Cochin to Kerala ( 1956 ) . The most notable recent exceptions
are Indian English spelling - changes of Orissa to Odisha ( March 2011 ) and the Union Territory of Pondicherry ( which includes the City of Pondicherry ) to
Puducherry .
</item 6>

<item 7>
Renaming of cities in India
Many people argue that such confusion can lead to indeterminate and / or unintended consequences See also ( edit ) India portal List of renamed Indian cities
and states Renaming in South Africa References ( edit ) Jump up ^ India and the World Bank : The Politics of Aid and Influence - Page 126 Jason A. Kirk - 2011
`` Orissa ( Note : This state was officially renamed Odisha in March 2011 ) '' Jump up ^ The Hindu Nationalist Movement and Indian Politics , 1925 to the 1990s
. - Page 134 Christophe Jaffrelot 1999 - `` The new state included Madhya Bharat , the Bhopal region , the former Vindhya Pradesh , Mahakoshal and Chhattisgarh
( the last two regions forming the Hindi - speaking parts in the former Madhya Pradesh ; see map , pp. xxii - xxiii ) . '' Jump up ^ Mira Kamdar Planet India :
How the Fastest Growing Democracy Is Transforming ... 2007 Author 's introduction Page xi `` India 's information - technology capital 's new name , should it
be adopted , will mean `` town of boiled beans . ''
</item 7>

<item 8>
Renaming of cities in India
In the early years after Indian independence , many name changes were effected in northern India for English spellings of Hindi place names that had simply been
romanised inconsistently by the British administration -- such as the British spelling `` Jubbulpore , '' renamed `` Jabalpur '' ( जबलपुर ) among the first
changes in 1947 . These changes did not generate significant controversy . More recent and high - profile changes -- including renaming such major cities as
Calcutta to Kolkata -- have generated greater controversy . Since independence , such changes have typically been enacted officially by legislation at local or
national Indian government level , and may or may not then be adopted by the Indian media , particularly the influential Indian press . In the case of smaller
towns and districts which were less notable outside and inside India , and where a well known English name ( or exonym ) could not be said to exist , older
spellings used under British India may not have had any specific legislation other than changes in practice on the romanisation of indigenous Indian language
names .
</item 8>

<item 9>
Renaming of cities in India
Jump up ^ `` Goa to become Govapuri '' . Jump up ^ `` Nagaland to be renamed '' . Aaj Tak . Jump up ^
http://www.dailypioneer.com/state-editions/lucknow/allahabad-all-set-to-get-new-name.html Jump up ^ TNN , Dec 18 , 2006 , 02.42 am IST ( 2006 - 12 - 18 ) . ``
Now , Indore to become Indur , Bhopal Bhojpal - The Times of India '' . The Times of India . Retrieved 2010 - 11 - 04 . CS1 maint : Multiple names : authors
list ( link ) Jump up ^ http://www.igovernment.in/opinion/41631/merging-npr-uid?source=igov External links ( edit ) Renaming roads : A ' meaningless ' exercise
12 Cities in Karnataka get a name change India 's Bangalore in name change Shashi Tharror : Becoming Bengaloorued Retrieved from ``
https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Renaming of cities in India
It is an indigenizing toponymic '' Jump up ^ Reserve Bank of India 's instructions for banks & banking operations Reserve Bank of India 2001 Page 713 `` The new
name `` Mumbai '' should be reflected in both English and Hindi and the change in name is to be brought about in all official communications , name plates ,
sign boards , office seals , rubber stamps , etc . '' Jump up ^ Perveez Mody The Intimate State : Love - Marriage and the Law in Delhi Page 59 - 2008 ``
Throughout this book , I refer to India 's commercial capital as Bombay rather than Mumbai ... I am well aware of the name - change effected by an Act of the
Indian Parliament in 1997 that made the city officially ' Mumbai ' ... It is the same convention I adopt when referring to Calcutta rather than Kolkata . ''
Jump up ^ Pingali Sailaja Indian English Page 16 2009 `` Bombay is now called Mumbai , Madras is now Chennai and Calcutta is Kolkata , in an attempt to
de-anglicise them . In this work , the earlier names are retained since these names were used during the period that we mostly cover . ''
</item 10>
=== END ITEM id=9bd159f1312738c0 ===

=== ITEM id=ad000ff0b36c4c12 ===
Gold sentences and facts:
Sentence 0: William Alan `` Will '' Friedle ( born August 11 , 1976 ) is an American actor and voice actor . He is best known for his comedic roles , most notably the underachieving elder brother Eric Matthews on the long - running TV sitcom Boy Meets World from 1993 to 2000 . More recently , he has voiced a number of animated characters such as Terry McGinnis / Batman , the title character of Batman Beyond , and Ron Stoppable of Kim Possible . He voices Deadpool in Ultimate Spider - Man and Star - Lord in the animated version of Guardians of the Galaxy , replacing Chris Cox . He also performed the voices of Doyle in The Secret Saturdays , Lion - O in the rebooted ThunderCats series , and Blue Beetle on Batman : The Brave and the Bold . One of his most recent roles has been the speaking voice of Bumblebee in the final episode of Transformers : Prime and the movie Transformers Prime Beast Hunters : Predacons Rising and in the sequel series Transformers : Robots in Disguise , as well as in Transformers : Rescue Bots .
  Fact 0.0: William Alan "Will" Friedle was born on August 11, 1976.
  Fact 0.1: William Alan "Will" Friedle is an American actor.
  Fact 0.2: William Alan "Will" Friedle is an American voice actor.
  Fact 0.3: Will Friedle is best known for comedic roles.
  Fact 0.4: Will Friedle is most notably known for playing the underachieving elder brother Eric Matthews on the long-running TV sitcom Boy Meets World.
  Fact 0.5: Will Friedle played Eric Matthews on Boy Meets World from 1993 to 2000.
  Fact 0.6: Will Friedle has more recently voiced a number of animated characters.
  Fact 0.7: Will Friedle voiced Terry McGinnis / Batman, the title character of Batman Beyond.
  Fact 0.8: Will Friedle voiced Ron Stoppable of Kim Possible.
  Fact 0.9: Will Friedle voices Deadpool in Ultimate Spider-Man.
  Fact 0.10: Will Friedle voices Star-Lord in the animated version of Guardians of the Galaxy.
  Fact 0.11: Will Friedle replaced Chris Cox in the animated Guardians of the Galaxy role.
  Fact 0.12: Will Friedle performed the voice of Doyle in The Secret Saturdays.
  Fact 0.13: Will Friedle performed the voice of Lion-O in the rebooted ThunderCats series.
  Fact 0.14: Will Friedle performed the voice of Blue Beetle on Batman: The Brave and the Bold.
  Fact 0.15: One of Will Friedle's most recent roles was the speaking voice of Bumblebee in the final episode of Transformers: Prime.
  Fact 0.16: Will Friedle was the speaking voice of Bumblebee in the movie Transformers Prime Beast Hunters: Predacons Rising.
  Fact 0.17: Will Friedle was the speaking voice of Bumblebee in the sequel series Transformers: Robots in Disguise.
  Fact 0.18: Will Friedle was the speaking voice of Bumblebee in Transformers: Rescue Bots.

Retrieved items:
<item 1>
Daren Kagasoff
Daren Kagasoff - wikipedia Daren Kagasoff Jump to : navigation , search Daren Kagasoff At the Twilight premiere Daren Maxwell Kagasoff ( 1987 - 09 - 16 )
September 16 , 1987 ( age 30 ) Encino , California , U.S. Occupation Actor Years active 2008 -- present Daren Maxwell Kagasoff ( KA - guh - sawf ) ( born
September 16 , 1987 ) is an American actor . He is known for starring as Ricky Underwood on the ABC Family teen drama series The Secret Life of the American
Teenager from 2008 to 2013 . Contents ( hide ) 1 Early life 2 Career 3 Filmography 4 Awards and nominations 5 References 6 External links Early Life ( edit )
Kagasoff was born in Encino , California to Elise Kagasoff ( née Levy ) and Barry Kagasoff , a diamond merchant . He is the middle of three children ; he has an
older brother , Justin , and a younger sister , Natalie .
</item 1>

<item 2>
Will Friedle
Will Friedle - wikipedia Will Friedle Jump to : navigation , search Will Friedle William Alan Friedle ( 1976 - 08 - 11 ) August 11 , 1976 ( age 41 ) Hartford ,
Connecticut , U.S. Occupation Actor , voice actor Years active 1988 -- present Spouse ( s ) Susan Martens ( m . 2016 ) William Alan `` Will '' Friedle ( born
August 11 , 1976 ) is an American actor and voice actor . He is best known for his comedic roles , most notably the underachieving elder brother Eric Matthews
on the long - running TV sitcom Boy Meets World from 1993 to 2000 . More recently , he has voiced a number of animated characters such as Terry McGinnis /
Batman , the title character of Batman Beyond , and Ron Stoppable of Kim Possible . He voices Deadpool in Ultimate Spider - Man and Star - Lord in the animated
version of Guardians of the Galaxy , replacing Chris Cox . He also performed the voices of Doyle in The Secret Saturdays , Lion - O in the rebooted ThunderCats
series , and Blue Beetle on Batman : The Brave and the Bold .
</item 2>

<item 3>
Will Friedle
Career ( edit ) After co-hosting Nickelodeon 's Do n't Just Sit There for three seasons ( 1988 -- 1991 ) , Friedle was cast as Cory Matthews ' likeable but
underachieving elder brother Eric on the long - running TV sitcom Boy Meets World from 1993 to 2000 . In 2004 , he co-starred with Chris Owen in the independent
film National Lampoon 's Gold Diggers ( originally titled Lady Killers ) . He also appeared in a supporting role in the 2005 ABC Family movie Everything You
Want . Friedle was cast as Mike in the WB sitcom Off Centre , but could not accept the role because of a previous commitment to the UPN series The Random Years
. The role ended up going to Eddie Kaye Thomas . Friedle voiced Seifer Almasy in the English version of the video game Kingdom Hearts II , Gideon Wyeth in the
English version of the video game Advent Rising , and Jaster Rogue in the English version of the video game Rogue Galaxy .
</item 3>

<item 4>
Will Friedle
Personal life ( edit ) In 1997 , Friedle dated American actress Jennifer Love Hewitt , and co-starred with her in the romantic teen comedy Trojan War . Hewitt
made a guest appearance as Jennifer Love `` Feffy '' Fefferman on Boy Meets World episode `` And Then There Was Shawn '' . Friedle is best friends with actor
Jason Marsden and was the best man at Marsden 's wedding in October 2004 . The two have also worked together in numerous productions : Boy Meets World , Trojan
War , Static Shock , Kim Possible , Batman Beyond , Justice League Unlimited , Batman : The Brave and the Bold , ThunderCats and Mad . He married his girlfriend
, Susan Martens , on September 25 , 2016 in Connecticut . Filmography ( edit ) Film Year Film Role Notes Trojan War Brad Kimble 2000 Batman Beyond : Return of
the Joker Terry McGinnis / Batman Voice role Direct - to - DVD release 2003 National Lampoon 's Gold Diggers Calvin Menhoffer The Greatest Short Film Ever ! ! !
</item 4>

<item 5>
Will Friedle
! ! Mark Short film Howl 's Moving Castle Additional voices Voice role 2005 Everything You Want Calvin Dillwaller 2008 Batman : Gotham Knight Anton , Cultist ,
Youth # 1 Voice role Direct - to - DVD release 2015 Batman Unlimited : Animal Instincts Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman
Unlimited : Monster Mayhem Nightwing / Dick Grayson Voice role Direct - to - DVD release 2016 Lego DC Comics Super Heroes : Justice League : Gotham City
Breakout Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman Unlimited : Mech vs. Mutants Nightwing / Dick Grayson Voice role Direct - to - DVD
release Television Year Title Role Notes Do n't Just Sit There Host Law & Order Russ Episode : `` Trust '' 1993 -- 2000 Boy Meets World Eric Matthews Series
regular ( 157 episodes ) Young Artist Award ( nominated ) The Gift of Love Luke CBS television movie Are You Afraid of the Dark ?
</item 5>

<item 6>
High School Musical 2
Taylor McKessie ( Monique Coleman ) is Gabriella 's best friend . She is also friends with Kelsi Nielsen and Martha Cox and is dating Chad . She is captain of
the school Scholastic Decathlon team . She has a summer job at Lava Springs as an Activities Coordinator . Thomas Fulton ( Mark L. Taylor ) is the manager of
Lava Springs . Coach Jack Bolton ( Bart Johnson ) is Troy 's father and East High 's basketball coach . Darby Evans ( Jessica Tuck ) is Sharpay and Ryan 's
mother and president of the Lava Springs Board . Vance Evans ( Robert Curtis Brown ) is Sharpay and Ryan 's father . Ms. Darbus ( Alyson Reed ) is the stern
drama teacher at East High . Zeke Baylor ( Chris Warren Jr . ) is friends with Troy and Chad and plays for the basketball team . Jason Cross ( Ryne Sanborn ) is
friends with Troy , Chad , and Zeke and plays on the basketball team . Kelsi Nielsen ( Olesya Rulin ) is friends with Gabriella and Taylor and a pianist and
composer .
</item 6>

<item 7>
Will Friedle
Kiel ( 28 January 2009 ) . `` The Bravest & Boldest : Will Friedle '' . Comic Book Resources . Retrieved 30 November 2010 . Jump up ^ Andy Swift . `` ' Girl
Meets World ' : Eric Returns With Mr. Feeny -- Season 2 Spoilers - TVLine '' . TVLine . Retrieved 4 April 2016 . Jump up ^ `` Girl Meets Writers '' . Twitter .
Retrieved 4 April 2016 . Jump up ^ `` Will Friedle on Twitter '' . Twitter . Retrieved 4 April 2016 . Jump up ^ `` Art - World Movie Shot In State To Be Shown
Saturday In New Milford '' . tribunedigital - thecourant . Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - Friedle - Leary '' .
Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - She Chose to Have Both '' . Retrieved 4 April 2016 .
</item 7>

<item 8>
Will Friedle
One of his most recent roles has been the speaking voice of Bumblebee in the final episode of Transformers : Prime and the movie Transformers Prime Beast
Hunters : Predacons Rising and in the sequel series Transformers : Robots in Disguise , as well as in Transformers : Rescue Bots . In December 2013 , it was
announced that Friedle would reprise his Boy Meets World role of Eric Matthews on the continuation series Girl Meets World . Contents ( hide ) 1 Early life 2
Career 3 Personal life 4 Filmography 5 References 6 External links Early life ( edit ) Friedle was born in Hartford , Connecticut , the son of Patricia Joan (
née Leary ) and Gary Allen Friedle , both lawyers . After committing himself to becoming an actor , Friedle continually commuted from Avon , Connecticut to New
York City for auditions . He attended Avon High School in 1994 . Friedle attended Occidental College . In the July 4 , 2006 issue of People , Friedle mentioned
he had thought about attending culinary school .
</item 8>

<item 9>
Will Friedle
Retrieved 4 April 2016 . Jump up ^ Friedle lands UPN sitcom Neil Wilkes , DigitalSpy , 4 March 2002 Jump up ^ `` Will Friedle '' . Retrieved 4 April 2016 . Jump
up ^ `` Boy Meets World star Will Friedle celebrates his marriage '' . Retrieved 15 August 2017 . Jump up ^ `` Gary Miereanu '' . Twitter . Retrieved 4 April
2016 . Jump up ^ Gerding , Stephen ( 2 June 2015 ) . `` EXCLUSIVE TRAILER DEBUT : Joker Rules in `` Batman Unlimited : Monster Mayhem '' `` . Comic Book
Resources . Retrieved 6 June 2015 . Jump up ^ Siegel , Lucas ( April 4 , 2016 ) . `` EXCLUSIVE : WB Reveals Next LEGO DC Comics Movie with Announcement Trailer
'' . Comic Book Resources . Retrieved July 25 , 2016 . Jump up ^ `` New Frontiers '' . Avengers Assemble . Season 2 . Episode 25 . September 13 , 2015 .
</item 9>

<item 10>
Daren Kagasoff
The film was released in October 2014 . In September 2014 , Kagasoff joined the cast of the Red Band Society , a comedy - drama series on Fox for the 2014 -- 15
American television season developed by Margaret Nagle . He had a recurring role playing Kara 's love interest , Hunter Cole . The series premiered on September
17 , 2014 and was canceled after one season . Filmography ( edit ) Television and film roles Year Title Role Notes 2008 -- 2013 Secret Life of the American
Teenager , The The Secret Life of the American Teenager Richard `` Ricky '' Underwood Main role 2014 Ouija Trevor Film 2014 Blue Daren Web series ; 2 episodes
2014 Delirium Alex Sheathes Unsold television pilot ( FOX ) 2014 Stalker Eric Bates Episode : Pilot 2014 -- 2015 Red Band Society Hunter Cole Recurring role , 7
episodes 2015 Paradise Pictures Wyatt Osborne Unsold television pilot ( USA ) 2017 S.W.A.T.
</item 10>
=== END ITEM id=ad000ff0b36c4c12 ===

=== ITEM id=f7bcd740219069f8 ===
Gold sentences and facts:
Sentence 0: The theory is generally associated with Hirschman . He presented a complete theoretical formulation of the strategy . Underdeveloped countries display common characteristics : low levels of GNI per capita and slow GNI per capita growth , large income inequalities and widespread poverty , low levels of productivity , great dependence on agriculture , a backward industrial structure , a high proportion of consumption and low savings , high rates of population growth and dependency burdens , high unemployment and underemployment , technological backwardness and dualism ( existence of both traditional and modern sectors ) . In a less - developed country , these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources . With a lack of investors and entrepreneurs , cash flows can not be directed into various sectors that influence balanced economic growth .
  Fact 0.0: The theory is generally associated with Hirschman.
  Fact 0.1: Hirschman presented a complete theoretical formulation of the strategy.
  Fact 0.2: Underdeveloped countries display common characteristics.
  Fact 0.3: Underdeveloped countries have low levels of GNI per capita.
  Fact 0.4: Underdeveloped countries have slow GNI per capita growth.
  Fact 0.5: Underdeveloped countries have large income inequalities.
  Fact 0.6: Underdeveloped countries have widespread poverty.
  Fact 0.7: Underdeveloped countries have low levels of productivity.
  Fact 0.8: Underdeveloped countries have great dependence on agriculture.
  Fact 0.9: Underdeveloped countries have a backward industrial structure.
  Fact 0.10: Underdeveloped countries have a high proportion of consumption and low savings.
  Fact 0.11: Underdeveloped countries have high rates of population growth and dependency burdens.
  Fact 0.12: Underdeveloped countries have high unemployment and underemployment.
  Fact 0.13: Underdeveloped countries have technological backwardness.
  Fact 0.14: Underdeveloped countries have dualism, meaning both traditional and modern sectors exist.
  Fact 0.15: In a less-developed country, these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources.
  Fact 0.16: In a less-developed country, a lack of investors and entrepreneurs means cash flows cannot be directed into various sectors that influence balanced economic growth.

Retrieved items:
<item 1>
Strategy of unbalanced growth
Therefore , growth need not take place in a balanced way . Supporters of the unbalanced growth doctrine include Albert O. Hirschman , Hans Singer , Paul
Streeten , Marcus Fleming , Prof. Rostov and J. Sheehan . Contents ( hide ) 1 Introduction 2 Complementarity 3 Induced investment 4 External economies 5 Social
Overhead Capital 6 Development via capital imbalances 7 Backward and forward linkages 8 Linkages and last industries 8.1 Disadvantages 9 Further reading 10
Notes 11 References Introduction ( edit ) The theory is generally associated with Hirschman . He presented a complete theoretical formulation of the strategy .
Underdeveloped countries display common characteristics : low levels of GNI per capita and slow GNI per capita growth , large income inequalities and widespread
poverty , low levels of productivity , great dependence on agriculture , a backward industrial structure , a high proportion of consumption and low savings ,
high rates of population growth and dependency burdens , high unemployment and underemployment , technological backwardness and dualism ( existence of both
traditional and modern sectors ) .
</item 1>

<item 2>
Economy of India
83 -- 85 . Jump up ^ Paul Bairoch ( 1995 ) . Economics and World History : Myths and Paradoxes . University of Chicago Press . pp. 95 -- 104 . Jump up ^ Chris
Jochnick , Fraser A. Preston ( 2006 ) , Sovereign Debt at the Crossroads : Challenges and Proposals for Resolving the Third World Debt Crisis , pages 86 - 87 ,
Oxford University Press ^ Jump up to : Parthasarathi , Prasannan ( 2011 ) , Why Europe Grew Rich and Asia Did Not : Global Economic Divergence , 1600 -- 1850 ,
Cambridge University Press , pp. 38 -- 45 , ISBN 978 - 1 - 139 - 49889 - 0 Jump up ^ Kumar 2005 , p. 422 Jump up ^ Maddison , Angus ( 2003 ) : Development
Centre Studies The World Economy Historical Statistics : Historical Statistics , OECD Publishing , ISBN 9264104143 , page 261 Jump up ^ James Cypher ( 2014 ) .
The Process of Economic Development . Routledge .
</item 2>

<item 3>
Hirschman’s unbalanced-growth strategy argues that developing economies should not aim for simultaneous, balanced investment. Selective investment creates
shortages and tensions that spur complementary and induced investment, while projects generate external economies. Development can begin with direct productive
activities (DPA), which pressure governments to expand social overhead capital (SOC), or with SOC, which attracts private investment. Prioritize sectors with
strong backward and forward linkages; import-processing “last industries” can launch industrialization with limited capital, but may
</item 3>

<item 4>
Strategy of unbalanced growth
In a less - developed country , these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources . With a lack of
investors and entrepreneurs , cash flows can not be directed into various sectors that influence balanced economic growth . Hirschman contends that deliberate
unbalancing of the economy according to the strategy is the best method of development and if the economy is to be kept moving ahead , the task of development
policy is to maintain tension , disproportions and disequilibrium . Balanced growth should not be the goal but rather the maintenance of existing imbalances ,
which can be seen from profit and losses . Therefore , the sequence that leads away from equilibrium is precisely an ideal pattern for development . Unequal
development of various sectors often generates conditions for rapid development . More - developed industries provide undeveloped industries an incentive to
grow . Hence , development of underdeveloped countries should be based on this strategy . The path of unbalanced growth is described by three phases :
Complementary Induced investment External economies Singer believed that desirable investment programs always exist within a country that represent unbalanced
investment to complement the existing imbalance . These investments create a new imbalance , requiring another balancing investment .
</item 4>

<item 5>
Strategy of unbalanced growth
( hide ) Development Economics : Partial Theories of Development Dual - sector model Big push model Critical minimum effort theory Ragnar Nurkse 's balanced
growth theory Strategy of unbalanced growth Social dualism Technological dualism Fei -- Ranis model of economic growth Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Strategy_of_unbalanced_growth&oldid=838124154 '' Categories : Economic growth Hidden categories : Wikipedia articles
that are incomprehensible from August 2016 All Wikipedia articles that are incomprehensible Articles lacking in - text citations from January 2012 All articles
lacking in - text citations All articles with unsourced statements Articles with unsourced statements from February 2013 Wikipedia articles needing
clarification from February 2013 Articles with unsourced statements from March 2012 All articles with specifically marked weasel - worded phrases Articles with
specifically marked weasel - worded phrases from February 2013 Talk Contents About Wikipedia Deutsch Français 中文 Edit links This page was last edited on 25
April 2018 , at 02 : 21 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 5>

<item 6>
Strategy of unbalanced growth
Strategy of unbalanced growth - Wikipedia Strategy of unbalanced growth Jump to : navigation , search This article may be incomprehensible or very hard to
understand . In particular , it uses both very general statements which are not further clarified or contextualized , and technical jargon that is not explained
. Please help by rewording it if the intended meaning can be determined . The talk page may have details . ( August 2016 ) ( Learn how and when to remove this
template message ) This article includes a list of references , related reading or external links , but its sources remain unclear because it lacks inline
citations . Please help to improve this article by introducing more precise citations . ( January 2012 ) ( Learn how and when to remove this template message )
Unbalanced growth is a natural path of economic development . Situations that countries are in at any one point in time reflect their previous investment
decisions and development . Accordingly , at any point in time desirable investment programs that are not balanced investment packages may still advance welfare
. Unbalanced investment can complement or correct existing imbalances . Once such an investment is made , a new imbalance is likely to appear , requiring
further compensating investments .
</item 6>

<item 7>
Strategy of unbalanced growth
The services provided should be subject to public control . The services can not be imported . The investment needed to provide services should be characterized
by some unevenness as well as by high capital output ratio . Development via capital imbalances ( edit ) The strategy of unbalanced growth has been discussed
within the frameworks of development through shortage of SOC and development through excess of SOC . In the first case , the country invests in direct
productive activities ( DPA ) . Direct productive activity increases demand for SOC , inducing investment . In the second case , SOC expands , which reduces the
cost of services , inducing investment in DPA . The cost of producing any unit of output of DPA is inversely proportional to SOC . The economy 's major
objective is to attain increasing output of DPA . One of the paradoxes of development is that poor countries can not afford to be economical . According to
Hirschman , resources are not scarce per se , but the ability to employ those resources may be lacking . To explain unbalanced growth , Hirschman assumes that
the country invests in either DPA or SOC .
</item 7>

<item 8>
Strategy of unbalanced growth
Paul Streeten , `` Unbalanced Growth . '' Economic Integration , Sythoff , Leiden ( Netherlands ) , 1961 . Reprinted in A.N. Agarwala and S.P. Singh , ( eds . )
, Accelerating Investment in Developing Economies ( London : Oxford University Press , 1969 ) .
</item 8>

<item 9>
Economy of India
Jump up ^ Datt & Sundharam 2009 , pp. 185 -- 187 Jump up ^ Cameron , John ; Ndhlovu , P Tidings ( September 2001 ) . `` Cultural Influences on Economic Thought
in India : Resistance to diffusion of neo-classical economics and the principles of Hinduism '' ( PDF ) . Economic Issues . 6 ( 2 ) . Archived from the original
( PDF ) on 23 August 2006 . Retrieved 17 January 2011 . Jump up ^ Roy , Subroto ( 22 September 1998 ) . `` Milton Friedman on the Nehru / Mahalanobis Plan '' .
India Policy Institute . Retrieved 16 July 2005 . Jump up ^ Panagariya 2008 , pp. 6 -- 7 Jump up ^ Tripathi , Salil ( 13 June 2006 ) . `` Escaping the ' Hindu
rate of growth ' '' . The Guardian . London . Retrieved 25 November 2010 . Jump up ^ Datt & Sundharam 2009 , pp.
</item 9>

<item 10>
Strategy of unbalanced growth
11 , No. 2 ( Jun. , 1959 ) , pp. 167 -- 190 Notes ( edit ) References ( edit ) Hirschman , Albert O . `` The Strategy of Economic Development , '' in Agarwal ,
A.N. and Singh , S.P. ( eds ) , Accelerating Investment in Developing Economies ( London Oxford Press , 1969 ) Thirwall , A.P. Growth and Development : With
special reference to developing economies ( London : Macmillian Press ltd ... , sixth edition . 1999 ) , pp237 -- 242 . Scitovsky , Tibor . Papers on welfare
and Growth ( London George Allen and Unwin ltd ... 1965 ) Refer to 5th paper entitled `` Growth -- Balanced and Unbalanced '' . Nath , S.K. , `` Balanced Growth
. '' in Livingstone ( ed ) , Economic Policy for Development ( Harmodworth , Middlesex : Penguin Books ltd. 1971 ) Han Singer , `` The Concept of Balanced
Growth and Economic Development ; Theory and Facts , '' University of Texas Conference on Economic Development , April 1958 .
</item 10>
=== END ITEM id=f7bcd740219069f8 ===
